"""
Premiere 工程输出：把保留片段变成一条视频和音频都剪好的序列。

.prproj 没有公开格式，里面还存着导入器 ID、文件状态等内部数据，手写很脆弱，所以分两步：
  1) 写一份 FCP7 XML（Premiere 支持的交换格式），视频和音频片段互相链接；
  2) 让本机 Premiere 导入这份 XML 并存成 .prproj，工程由 Premiere 自己写出，版本也完全匹配。
第 2 步做不了时（没装 Premiere、Premiere 正开着等）交付 XML，手动导入后结果相同。

序列里还标出值得人工先看的地方（checks，来自 dedup._check_points）：
  * 整行剪掉的（画外话、后面重说过的前一版）：在剪切处放一个标记，写着被剪掉的内容，恢复时把片段边缘拉回来；
  * 留下但存疑的：切成单独的片段并换一种标签颜色，上面放一个带时长的标记，确认该删时选中它波纹删除。
"""
import fractions, glob, json, os, re, shutil, subprocess, time, urllib.parse
from typing import Callable, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

EXE_NAME = "Adobe Premiere Pro.exe"
# Premiere 内部的时间单位：每秒的 tick 数
TICKS_PER_SECOND = 254016000000
# 太短的片段不进序列，下限与音频拼接相同
MIN_CLIP_SEC = 0.04
# 纯音频素材也要有序列帧率：帧率越高，切点取整的误差越小
AUDIO_ONLY_TIMEBASE = 60
# 存疑片段的标签颜色（Premiere 默认标签名）。用户改过标签名时不生效，片段保持默认颜色，标记仍然在
FLAG_LABEL = "Mango"
# 切出存疑片段时，两边剩下的片段不短于这么多，否则不在那里切（很短的片段进不了序列，会和音频文件对不上）
MIN_PIECE_SEC = 0.15
# 标记的说明里最多放这么多字的原文
MARKER_TEXT_CHARS = 300


class SourceUnreadable(RuntimeError):
    """Premiere 读不了这个素材（如 MKV）：序列里全是离线片段，交付什么都没用。"""


def _parse_rate(text) -> Optional[fractions.Fraction]:
    try:
        rate = fractions.Fraction(str(text))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _probe_media(path: str) -> Dict:
    """读取生成序列需要的素材参数。封面图（mp3/m4a 里的 attached_pic）不算视频。"""
    data = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video and not audio:
        raise RuntimeError("The source has no video or audio stream.")
    return {"video": video, "audio": audio, "duration": float(data["format"]["duration"]),
            "audio_offset": _audio_offset(audio, data.get("format") or {})}


def _audio_offset(audio: Optional[Dict], container: Dict) -> float:
    """音频流比文件起点晚多少秒开始（多数文件是 0）。

    保留区间是按解码出来的音频计时的：ffmpeg 解码成裸 PCM 时不会补上开头这段空白，第一个采样就是 0 秒。
    Premiere 按文件的时间放置素材，所以序列里的时间要加上这段偏移，否则每个切点都早这么多。
    """
    if not audio:
        return 0.0
    try:
        return max(0.0, float(audio.get("start_time")) - float(container.get("start_time")))
    except (TypeError, ValueError):
        return 0.0


def _timeline_rate(video: Optional[Dict]) -> Tuple[int, bool]:
    """素材帧率对应的序列帧率 (timebase, ntsc)。可变帧率素材用平均帧率。"""
    if not video:
        return AUDIO_ONLY_TIMEBASE, False
    rate = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or fractions.Fraction(30)
    value = float(rate)
    nominal = max(1, round(value))
    return nominal, abs(value - nominal * 1000 / 1001) < 0.02 and abs(value - nominal) > 0.02


def _split_at_flags(keep_blocks: List[Tuple[float, float]],
                    flag_spans: List[Tuple[float, float]]) -> List[Tuple[float, float, bool]]:
    """把保留区间在存疑片段的边界处切开，返回 [(起点, 终点, 是否存疑)]。

    切出来的片段首尾相接，合起来和原来的区间完全一样：只是多了几个剪辑点，播出来的内容不变。
    """
    pieces: List[Tuple[float, float, bool]] = []
    flags = sorted(flag_spans)
    for start, end in sorted(keep_blocks):
        position = start
        for flag_start, flag_end in flags:
            flag_start, flag_end = max(flag_start, position), min(flag_end, end)
            if flag_end - flag_start < MIN_PIECE_SEC:
                continue
            if flag_start - position < MIN_PIECE_SEC:
                flag_start = position
            if end - flag_end < MIN_PIECE_SEC:
                flag_end = end
            if flag_start > position:
                pieces.append((position, flag_start, False))
            pieces.append((flag_start, flag_end, True))
            position = flag_end
        if end > position:
            pieces.append((position, end, False))
    return pieces


def _frame_spans(pieces: List[Tuple[float, float, bool]], fps: fractions.Fraction,
                 total_frames: int) -> List[Tuple[int, int, int, int, bool]]:
    """把片段取整到帧，返回 (序列起点, 序列终点, 素材入点, 素材出点, 是否存疑)。片段首尾相接，不重复任何一帧。"""
    spans = []
    position = source_end = 0
    for start, end, flagged in sorted(pieces):
        if end - start <= MIN_CLIP_SEC:
            continue
        first = max(round(start * fps), source_end)
        last = min(round(end * fps), total_frames)
        if last <= first:
            continue
        spans.append((position, position + last - first, first, last, flagged))
        position += last - first
        source_end = last
    return spans


def _sequence_frame(spans: List[Tuple[int, int, int, int, bool]], source_frame: int) -> int:
    """素材里的一帧在序列里的位置。它被剪掉时，取它后面第一个保留片段的开头：那就是剪切处。"""
    for start, end, first, last, _ in spans:
        if source_frame < last:
            return start + max(0, source_frame - first)
    return spans[-1][1]


def _markers(checks: List[Dict], spans: List[Tuple[int, int, int, int, bool]], fps: fractions.Fraction,
             offset: float) -> List[Dict]:
    """把要人工先看的地方变成序列标记：name、comment、start / end（序列帧；点标记的 end 为 -1）。"""
    markers = []
    for check in checks:
        start = _sequence_frame(spans, round((check["start"] + offset) * fps))
        end = -1
        if check["kind"] == "flag":
            end = _sequence_frame(spans, round((check["end"] + offset) * fps))
            if end <= start:
                continue  # 这一段没有进序列（比一帧还短）
        text = check["text"] if len(check["text"]) <= MARKER_TEXT_CHARS else check["text"][:MARKER_TEXT_CHARS] + "…"
        markers.append({"name": ("Cut: " if check["kind"] == "cut" else "Check: ") + check["why"],
                        "comment": text, "start": start, "end": end})
    return markers


def write_fcp_xml(media_path: str, keep_blocks: List[Tuple[float, float]], xml_path: str, name: str,
                  checks: Optional[List[Dict]] = None) -> Tuple[int, List[Dict]]:
    """写 FCP7 XML，结构与 Premiere 自己导出的一致。返回 (序列里的片段数, 序列标记)。

    checks 是要人工先看的地方（见模块说明）。标记里的 start / end 另外带一份秒数（start_sec / end_sec），
    导入 XML 之后标记没有跟过来时，用它在 Premiere 里补建。
    """
    info = _probe_media(media_path)
    video, audio = info["video"], info["audio"]
    timebase, ntsc = _timeline_rate(video)
    fps = fractions.Fraction(timebase * 1000, 1001) if ntsc else fractions.Fraction(timebase)
    total_frames = int(round(info["duration"] * fps))
    ticks_per_frame = int(TICKS_PER_SECOND / fps)
    offset = info["audio_offset"]
    checks = checks or []
    pieces = _split_at_flags([(start + offset, end + offset) for start, end in keep_blocks],
                             [(c["start"] + offset, c["end"] + offset) for c in checks if c["kind"] == "flag"])
    spans = _frame_spans(pieces, fps, total_frames)
    if not spans:
        raise RuntimeError("No keep blocks for the Premiere sequence.")
    markers = _markers(checks, spans, fps, offset)
    for marker in markers:
        marker["start_sec"] = float(marker["start"] / fps)
        marker["end_sec"] = float(marker["end"] / fps) if marker["end"] > 0 else -1.0

    channels = int(audio.get("channels") or 2) if audio else 0
    stereo = channels == 2
    sample_rate = int(audio.get("sample_rate") or 48000) if audio else 48000
    clip_name = escape(os.path.basename(media_path))
    url = "file://localhost/" + urllib.parse.quote(os.path.abspath(media_path).replace("\\", "/"), safe="/")

    rate = "<rate><timebase>%d</timebase><ntsc>%s</ntsc></rate>" % (timebase, "TRUE" if ntsc else "FALSE")
    timecode = ("<timecode>%s<string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode>"
                % rate)
    width, height = (int(video["width"]), int(video["height"])) if video else (1920, 1080)
    video_format = ("<samplecharacteristics>%s<width>%d</width><height>%d</height><anamorphic>FALSE</anamorphic>"
                    "<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance>"
                    "</samplecharacteristics>" % (rate, width, height))
    audio_format = ("<samplecharacteristics><depth>16</depth><samplerate>%d</samplerate></samplecharacteristics>"
                    % sample_rate)
    media_parts = ""
    if video:
        media_parts += "<video>%s</video>" % video_format
    if audio:
        media_parts += "<audio>%s<channelcount>%d</channelcount></audio>" % (audio_format, channels)
    file_full = ('<file id="file-1"><name>%s</name><pathurl>%s</pathurl>%s<duration>%d</duration>%s<media>%s</media></file>'
                 % (clip_name, escape(url), rate, total_frames, timecode, media_parts))

    def _links(n: int) -> str:
        links = ""
        if video:
            links += ("<link><linkclipref>clipitem-v%d</linkclipref><mediatype>video</mediatype>"
                      "<trackindex>1</trackindex><clipindex>%d</clipindex></link>" % (n, n))
        for track in range(1, channels + 1):
            links += ("<link><linkclipref>clipitem-a%d-%d</linkclipref><mediatype>audio</mediatype>"
                      "<trackindex>%d</trackindex><clipindex>%d</clipindex>%s</link>"
                      % (track, n, track, n, "<groupindex>1</groupindex>" if stereo else ""))
        return links

    # 素材的完整定义只写在第一个片段里，其余片段按 id 引用
    file_written = [False]

    def _file_element() -> str:
        if file_written[0]:
            return '<file id="file-1"/>'
        file_written[0] = True
        return file_full

    def _clip(clip_id: str, attrs: str, media_type: str, source_track: int, n: int, span, extra: str = "") -> str:
        start, end, first, last, flagged = span
        label = "<labels><label2>%s</label2></labels>" % FLAG_LABEL if flagged else ""
        return ('<clipitem id="%s"%s><masterclipid>masterclip-1</masterclipid><name>%s</name><enabled>TRUE</enabled>'
                '<duration>%d</duration>%s<start>%d</start><end>%d</end><in>%d</in><out>%d</out>'
                '<pproTicksIn>%d</pproTicksIn><pproTicksOut>%d</pproTicksOut>%s%s'
                '<sourcetrack><mediatype>%s</mediatype><trackindex>%d</trackindex></sourcetrack>%s%s</clipitem>'
                % (clip_id, attrs, clip_name, total_frames, rate, start, end, first, last,
                   first * ticks_per_frame, last * ticks_per_frame, extra, _file_element(),
                   media_type, source_track, _links(n), label))

    video_track = ""
    if video:
        clips = "".join(_clip("clipitem-v%d" % (n + 1), "", "video", 1, n + 1, span,
                              "<alphatype>none</alphatype><pixelaspectratio>square</pixelaspectratio><anamorphic>FALSE</anamorphic>")
                        for n, span in enumerate(spans))
        video_track = "<track>%s<enabled>TRUE</enabled><locked>FALSE</locked></track>" % clips

    # 立体声和 Premiere 导出时一样拆成两条互相关联的轨；其他声道数每个声道一条单声道轨
    audio_tracks = ""
    for track in range(1, channels + 1):
        if stereo:
            track_attrs = ' currentExplodedTrackIndex="%d" totalExplodedTrackCount="2" premiereTrackType="Stereo"' % (track - 1)
            clip_attrs = ' premiereChannelType="stereo"'
        else:
            track_attrs = ' currentExplodedTrackIndex="0" totalExplodedTrackCount="1" premiereTrackType="Mono"'
            clip_attrs = ' premiereChannelType="mono"'
        clips = "".join(_clip("clipitem-a%d-%d" % (track, n + 1), clip_attrs, "audio", track, n + 1, span)
                        for n, span in enumerate(spans))
        audio_tracks += ("<track%s>%s<enabled>TRUE</enabled><locked>FALSE</locked><outputchannelindex>%d</outputchannelindex></track>"
                         % (track_attrs, clips, 2 - track % 2))

    outputs = "".join("<group><index>%d</index><numchannels>1</numchannels><downmix>0</downmix>"
                      "<channel><index>%d</index></channel></group>" % (i, i) for i in (1, 2))
    marker_items = "".join("<marker><comment>%s</comment><name>%s</name><in>%d</in><out>%d</out></marker>"
                           % (escape(m["comment"]), escape(m["name"]), m["start"], m["end"]) for m in markers)
    doc = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
           '<sequence id="sequence-1"><name>%s</name><duration>%d</duration>%s<in>-1</in><out>-1</out>%s'
           '<media><video><format>%s</format>%s</video>'
           '<audio><numOutputChannels>2</numOutputChannels><format>%s</format><outputs>%s</outputs>%s</audio></media>'
           '%s</sequence></xmeml>\n'
           % (escape(name), spans[-1][1], rate, timecode, video_format, video_track, audio_format, outputs, audio_tracks,
              marker_items))
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return len(spans), markers


def find_premiere() -> Optional[str]:
    """找本机 Premiere。可用环境变量 PREMIERE_EXE 指定；否则用安装目录里年份最新的正式版。"""
    configured = os.environ.get("PREMIERE_EXE")
    if configured and os.path.isfile(configured):
        return configured
    if os.name != "nt":
        return None
    base = os.environ.get("ProgramFiles") or r"C:\Program Files"
    candidates = {}
    for path in glob.glob(os.path.join(glob.escape(base), "Adobe", "Adobe Premiere Pro*", EXE_NAME)):
        year = re.search(r"(\d{4})", os.path.basename(os.path.dirname(path)))
        if year and "beta" not in path.lower():
            candidates[int(year.group(1))] = path
    return candidates[max(candidates)] if candidates else None


def premiere_running() -> bool:
    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq " + EXE_NAME, "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, errors="replace")
    return EXE_NAME.lower() in result.stdout.lower()


_IMPORT_SCRIPT = r"""
(function () {
  var XML_PATH = %(xml)s, PROJECT_PATH = %(project)s, STATUS_PATH = %(status)s, MARKERS = %(markers)s;
  function report(line) {
    var f = new File(STATUS_PATH);
    f.encoding = "UTF-8";
    f.open("a");
    f.writeln(line);
    f.close();
  }
  // Sequence markers come from the XML. If the import dropped them, create them here.
  // Whatever goes wrong only costs markers, never the project.
  function ensureMarkers(seq) {
    var have = -1, how = "imported";
    try {
      have = seq.markers.numMarkers;
      if (have === 0) {
        how = "created";
        for (var i = 0; i < MARKERS.length; i++) {
          try {
            var marker = seq.markers.createMarker(MARKERS[i].start_sec);
            marker.name = MARKERS[i].name;
            marker.comments = MARKERS[i].comment;
            if (MARKERS[i].end_sec > MARKERS[i].start_sec) { setMarkerEnd(marker, MARKERS[i].end_sec); }
          } catch (e) {
            report("MARKERS_ERR marker " + i + ": " + e);
          }
        }
        have = seq.markers.numMarkers;
      }
    } catch (e2) {
      report("MARKERS_ERR " + e2);
    }
    return have + " " + how;
  }
  // A marker's end takes seconds as a number in Premiere 2025 ("Illegal Parameter type" for a Time object).
  // Other versions document a Time object, so that is tried second. If both fail the marker stays a point.
  function setMarkerEnd(marker, seconds) {
    try {
      marker.end = seconds;
    } catch (e) {
      try {
        var end = new Time();
        end.seconds = seconds;
        marker.end = end;
      } catch (e2) {
        report("MARKERS_ERR end: " + e + " / " + e2);
      }
    }
  }
  try {
    if (!app.newProject(PROJECT_PATH)) { throw new Error("could not create the project"); }
    $.sleep(2000);
    if (!app.project.importFiles([XML_PATH], true, app.project.rootItem, false)) { throw new Error("could not import the XML"); }
    for (var i = 0; i < 240 && app.project.sequences.numSequences === 0; i++) { $.sleep(500); }
    if (app.project.sequences.numSequences === 0) { throw new Error("the XML produced no sequence"); }
    var seq = app.project.sequences[0];
    app.project.openSequence(seq.sequenceID);
    var video = seq.videoTracks.numTracks ? seq.videoTracks[0].clips.numItems : 0;
    var audio = seq.audioTracks.numTracks ? seq.audioTracks[0].clips.numItems : 0;
    report("CLIPS " + video + " " + audio);
    var firstClip = video ? seq.videoTracks[0].clips[0] : (audio ? seq.audioTracks[0].clips[0] : null);
    if (firstClip && firstClip.projectItem.isOffline()) { throw new Error("OFFLINE"); }
    if (MARKERS.length) { report("MARKERS " + ensureMarkers(seq)); }
    if (!app.project.save()) { throw new Error("could not save the project"); }
    $.sleep(1000);
    report("OK");
  } catch (e) {
    report("ERR " + e);
  }
  try { app.quit(); } catch (e2) { report("QUIT " + e2); }
})();
"""


def _read_status(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def _stop_process(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def convert_to_prproj(exe: str, xml_path: str, project_path: str, clip_count: int, has_video: bool,
                      has_audio: bool, work_dir: str, log: Callable[[str], None] = print,
                      markers: Optional[List[Dict]] = None) -> Optional[int]:
    """启动 Premiere，导入 XML 并存成 project_path，然后退出。失败时抛 RuntimeError，不留下半成品。

    markers 是 XML 里的序列标记（见 write_fcp_xml）。返回存好的序列里有几个标记；没有标记要加、
    或者 Premiere 没有报告数量时返回 None。
    调用前必须确认 Premiere 没在运行：/C es.processFile 对已打开的 Premiere 不起作用。
    """
    status_path = os.path.join(work_dir, "premiere_status.txt")
    script_path = os.path.join(work_dir, "premiere_import.jsx")
    # json.dumps 产出合法的脚本字符串，非 ASCII 字符转义后不受脚本文件编码影响
    paths = {key: json.dumps(value.replace("\\", "/"))
             for key, value in (("xml", xml_path), ("project", project_path), ("status", status_path))}
    paths["markers"] = json.dumps([{key: m[key] for key in ("name", "comment", "start_sec", "end_sec")}
                                   for m in markers or []])
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(_IMPORT_SCRIPT % paths)

    # Premiere 会往当前目录写日志，退出后辅助进程还会占着这个目录一会儿，
    # 所以用一个固定目录，不用 work_dir，否则任务结束时删不掉 work_dir
    cwd = os.path.join(os.path.dirname(os.path.abspath(work_dir)), "premiere")
    os.makedirs(cwd, exist_ok=True)
    proc = subprocess.Popen([exe, "/C", "es.processFile", script_path.replace("\\", "/")], cwd=cwd,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 180 + clip_count * 0.1
    lines: List[str] = []
    try:
        while time.time() < deadline:
            lines = _read_status(status_path)
            if any(line == "OK" or line.startswith("ERR ") for line in lines):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Premiere did not finish in time; it may be waiting on a dialog")
        log("premiere: %s" % " | ".join(lines))
        errors = [line[4:] for line in lines if line.startswith("ERR ")]
        if errors and "OFFLINE" in errors[0]:
            raise SourceUnreadable("Premiere cannot read the source file")
        if errors:
            raise RuntimeError(errors[0])
        counts = next((line.split() for line in lines if line.startswith("CLIPS ")), None)
        expected = ["CLIPS", str(clip_count if has_video else 0), str(clip_count if has_audio else 0)]
        if counts != expected:
            raise RuntimeError("the sequence has %s, expected %s" % (counts, expected))
        if not os.path.isfile(project_path):
            raise RuntimeError("Premiere did not write the project file")
    except Exception:
        _stop_process(proc)
        _remove_partial_project(project_path)
        raise
    # 工程已存好，这里只等 Premiere 自己退出；卡在退出阶段也不影响结果
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        log("premiere: did not exit in time, stopping it")
        _stop_process(proc)
    found = next((line.split() for line in lines if line.startswith("MARKERS ")), None)
    try:
        return int(found[1]) if found else None
    except (IndexError, ValueError):
        return None


def _remove_partial_project(project_path: str) -> None:
    """删掉没写完的工程，以及保存中断时 Premiere 留在旁边的 <工程名><GUID> 临时文件。"""
    guid = "-".join("?" * n for n in (8, 4, 4, 4, 12))
    stem = os.path.splitext(project_path)[0]
    for path in [project_path] + glob.glob(glob.escape(stem) + guid):
        try:
            os.remove(path)
        except OSError:
            pass


def export_project(media_path: str, keep_blocks: List[Tuple[float, float]], out_stem: str, work_dir: str,
                   log: Callable[[str], None] = print,
                   checks: Optional[List[Dict]] = None) -> Tuple[Optional[str], Optional[str]]:
    """输出 Premiere 工程，返回 (文件路径, 警告)。优先 .prproj，做不到时交付 .xml 并在警告里说明原因和做法。

    checks 是要人工先看的地方（见模块说明），在序列里做成标记和换了颜色的片段。
    Premiere 读不了素材时不交付工程（路径为 None）：导入 XML 也只会得到离线片段。
    """
    name = os.path.basename(out_stem)
    xml_tmp = os.path.join(work_dir, name + ".xml")
    clip_count, markers = write_fcp_xml(media_path, keep_blocks, xml_tmp, name, checks)
    info = _probe_media(media_path)
    if info["audio_offset"] >= 0.001:
        log("premiere: the audio starts %.3f s into the file, so the sequence times are shifted by that much"
            % info["audio_offset"])
    project_path = out_stem + ".prproj"

    exe = find_premiere()
    if not exe:
        reason = "Adobe Premiere Pro was not found (set PREMIERE_EXE if it is installed elsewhere)"
    elif premiere_running():
        reason = "Premiere is open, and it can only be driven when it starts fresh. Close it before the next run to get a .prproj"
    else:
        try:
            found = convert_to_prproj(exe, xml_tmp, project_path, clip_count, bool(info["video"]),
                                      bool(info["audio"]), work_dir, log, markers)
            if markers and (found is None or found <= 0):
                return project_path, ("The Premiere project was created, but the %d markers for the passages to check "
                                      "could not be added. The same places are listed at the top of the cut list."
                                      % len(markers))
            return project_path, None
        except SourceUnreadable:
            return None, ("No Premiere project was created: Premiere cannot read this file format (MKV is a common case). "
                          "Convert it to MP4 first, e.g. ffmpeg -i input.mkv -c copy output.mp4, then run on that file.")
        except Exception as e:
            log("[warn] premiere project failed: %r" % str(e)[:300])
            reason = "Premiere could not create it: %s" % str(e)[:200]

    xml_path = out_stem + ".xml"
    shutil.move(xml_tmp, xml_path)
    return xml_path, "No .prproj was created. %s. Import the .xml instead (File > Import in Premiere): it gives the same sequence." % reason
