"""
Premiere 工程输出：把保留片段变成一条视频和音频都剪好的序列。

.prproj 没有公开格式，里面还存着导入器 ID、文件状态等内部数据，手写很脆弱，所以分两步：
  1) 写一份 FCP7 XML（Premiere 支持的交换格式），视频和音频片段互相链接；
  2) 让本机 Premiere 导入这份 XML 并存成 .prproj，工程由 Premiere 自己写出，版本也完全匹配。
第 2 步做不了时（没装 Premiere、Premiere 正开着等）交付 XML，手动导入后结果相同。
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
    return {"video": video, "audio": audio, "duration": float(data["format"]["duration"])}


def _timeline_rate(video: Optional[Dict]) -> Tuple[int, bool]:
    """素材帧率对应的序列帧率 (timebase, ntsc)。可变帧率素材用平均帧率。"""
    if not video:
        return AUDIO_ONLY_TIMEBASE, False
    rate = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or fractions.Fraction(30)
    value = float(rate)
    nominal = max(1, round(value))
    return nominal, abs(value - nominal * 1000 / 1001) < 0.02 and abs(value - nominal) > 0.02


def _frame_spans(keep_blocks: List[Tuple[float, float]], fps: fractions.Fraction,
                 total_frames: int) -> List[Tuple[int, int, int, int]]:
    """把保留区间取整到帧，返回 (序列起点, 序列终点, 素材入点, 素材出点)。片段首尾相接，不重复任何一帧。"""
    spans = []
    position = source_end = 0
    for start, end in sorted(keep_blocks):
        if end - start <= MIN_CLIP_SEC:
            continue
        first = max(round(start * fps), source_end)
        last = min(round(end * fps), total_frames)
        if last <= first:
            continue
        spans.append((position, position + last - first, first, last))
        position += last - first
        source_end = last
    return spans


def write_fcp_xml(media_path: str, keep_blocks: List[Tuple[float, float]], xml_path: str, name: str) -> int:
    """写 FCP7 XML，结构与 Premiere 自己导出的一致。返回序列里的片段数。"""
    info = _probe_media(media_path)
    video, audio = info["video"], info["audio"]
    timebase, ntsc = _timeline_rate(video)
    fps = fractions.Fraction(timebase * 1000, 1001) if ntsc else fractions.Fraction(timebase)
    total_frames = int(round(info["duration"] * fps))
    ticks_per_frame = int(TICKS_PER_SECOND / fps)
    spans = _frame_spans(keep_blocks, fps, total_frames)
    if not spans:
        raise RuntimeError("No keep blocks for the Premiere sequence.")

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
        start, end, first, last = span
        return ('<clipitem id="%s"%s><masterclipid>masterclip-1</masterclipid><name>%s</name><enabled>TRUE</enabled>'
                '<duration>%d</duration>%s<start>%d</start><end>%d</end><in>%d</in><out>%d</out>'
                '<pproTicksIn>%d</pproTicksIn><pproTicksOut>%d</pproTicksOut>%s%s'
                '<sourcetrack><mediatype>%s</mediatype><trackindex>%d</trackindex></sourcetrack>%s</clipitem>'
                % (clip_id, attrs, clip_name, total_frames, rate, start, end, first, last,
                   first * ticks_per_frame, last * ticks_per_frame, extra, _file_element(),
                   media_type, source_track, _links(n)))

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
    doc = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
           '<sequence id="sequence-1"><name>%s</name><duration>%d</duration>%s<in>-1</in><out>-1</out>%s'
           '<media><video><format>%s</format>%s</video>'
           '<audio><numOutputChannels>2</numOutputChannels><format>%s</format><outputs>%s</outputs>%s</audio></media>'
           '</sequence></xmeml>\n'
           % (escape(name), spans[-1][1], rate, timecode, video_format, video_track, audio_format, outputs, audio_tracks))
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return len(spans)


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
  var XML_PATH = %(xml)s, PROJECT_PATH = %(project)s, STATUS_PATH = %(status)s;
  function report(line) {
    var f = new File(STATUS_PATH);
    f.encoding = "UTF-8";
    f.open("a");
    f.writeln(line);
    f.close();
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
                      has_audio: bool, work_dir: str, log: Callable[[str], None] = print) -> None:
    """启动 Premiere，导入 XML 并存成 project_path，然后退出。失败时抛 RuntimeError，不留下半成品。

    调用前必须确认 Premiere 没在运行：/C es.processFile 对已打开的 Premiere 不起作用。
    """
    status_path = os.path.join(work_dir, "premiere_status.txt")
    script_path = os.path.join(work_dir, "premiere_import.jsx")
    # json.dumps 产出合法的脚本字符串，非 ASCII 字符转义后不受脚本文件编码影响
    paths = {key: json.dumps(value.replace("\\", "/"))
             for key, value in (("xml", xml_path), ("project", project_path), ("status", status_path))}
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
                   log: Callable[[str], None] = print) -> Tuple[Optional[str], Optional[str]]:
    """输出 Premiere 工程，返回 (文件路径, 警告)。优先 .prproj，做不到时交付 .xml 并在警告里说明原因和做法。

    Premiere 读不了素材时不交付工程（路径为 None）：导入 XML 也只会得到离线片段。
    """
    name = os.path.basename(out_stem)
    xml_tmp = os.path.join(work_dir, name + ".xml")
    clip_count = write_fcp_xml(media_path, keep_blocks, xml_tmp, name)
    info = _probe_media(media_path)
    project_path = out_stem + ".prproj"

    exe = find_premiere()
    if not exe:
        reason = "Adobe Premiere Pro was not found (set PREMIERE_EXE if it is installed elsewhere)"
    elif premiere_running():
        reason = "Premiere is open, and it can only be driven when it starts fresh. Close it before the next run to get a .prproj"
    else:
        try:
            convert_to_prproj(exe, xml_tmp, project_path, clip_count, bool(info["video"]), bool(info["audio"]), work_dir, log)
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
