"""
音频自动去重引擎：
  1) CrisperWhisper 逐字转写（在独立解释器中运行，带词级时间戳）
  2) 规则清理：括号声音标记、无歧义语气词（精确、无需模型）
  3) LLM（OpenAI 兼容，chat.completions / responses 端点）逐词判定口误，返回要删除的词索引
  4) ffmpeg 流式解码 → 只保留选中区间 → 流式编码，内存占用与时长无关
  5) 输出视频和音频都剪好的 Premiere 工程（见 premiere.py）
"""
import bisect, collections, datetime, json, os, re, shutil, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# Windows 中文环境 stdout 默认 GBK，遇到非 GBK 字符会抛 UnicodeEncodeError；强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from typing import List, Dict, Optional, Tuple, Set

import numpy as np

import premiere

# 让模型/依赖的缓存落在工作区内（Windows 沙箱限制 C:\Users 下缓存写入）
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, ".hf-cache"))


# 必然删除的无意义语气词（无歧义，直接删；其余交给 LLM 判断）
CORE_FILLERS = {"um", "uh", "erm", "er", "em", "mm", "hmm", "ah", "oh", "uhh", "eh"}
# 中文口癖/感叹语气词。转写可能按单字切分，所以这里只放几乎不会构成其他词的字；
# 哈（哈佛）、喂（喂养）、吓（吓人）、呵（呵护）、呐（呐喊）这类有歧义的字，
# 以及“那个/就是/然后”这类有语义的连接词，都交给 LLM 按上下文判断
CORE_FILLERS_ZH = {
    "嗯", "唔", "呃", "啊", "哦", "噢", "咦", "诶", "欸", "哎", "唉", "喔",
    "哎呀", "哎哟", "嗯嗯", "啊啊", "哦哦", "嘻嘻",
}
# 中文语言码识别
_ZH_LANGS = {"zh", "zh-cn", "zh-tw", "yue", "chi", "chinese"}


def _core_fillers_for(language: str) -> Set[str]:
    """按语言返回「必然删除」的语气词集合。"""
    lang = str(language or "").strip().lower()
    return CORE_FILLERS_ZH if lang in _ZH_LANGS else CORE_FILLERS


def _norm_language(language: str) -> str:
    """把语言码归一化成 Whisper 接受的 ISO 码：中文归 'zh'，其余归 'en'。"""
    lang = str(language or "").strip().lower()
    return "zh" if lang in _ZH_LANGS else "en"


def _is_bracket_sound(w: str) -> bool:
    """形如 [throatclearing]、[UM]、[cough]、【咳嗽】 之类的声音标记。"""
    return bool(re.match(r"^\s*(\[[^\]]+\]|【[^】]+】)", w))


def _report(progress, frac: float, msg: str):
    if progress:
        try:
            progress(frac, msg)
        except Exception:
            pass


# 中间文件和日志都放在 .work，output 里只留最终结果
WORK_DIR = os.path.join(_HERE, ".work")
_LOG_FILE = os.path.join(WORK_DIR, "debug.log")
_dbg_lock = threading.Lock()
# 管道每次读写的大小：拼接时的内存占用由它决定，与音频时长无关
_PIPE_CHUNK = 1 << 20


def new_job_dir() -> str:
    """为一次处理新建中间文件目录，调用方负责在结束时删除。"""
    os.makedirs(WORK_DIR, exist_ok=True)
    return tempfile.mkdtemp(prefix="job_", dir=WORK_DIR)


def _dbg(msg):
    """打印并写入 debug 日志，便于诊断 LLM 输出 / 区间 / 时长问题。"""
    line = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with _dbg_lock:
            os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


def transcribe(audio_path: str, model_size: str = "large", language: str = "en", progress=None,
               *, work_dir: str) -> List[Dict]:
    """用 CrisperWhisper verbatim 转写（全局 Python 3.14 + torch），返回词级 token 列表。

    子进程分块转写并实时把 STAGE/PROGRESS 打到 stdout，本函数用 Popen + 读线程
    解析并转给 UI，避免长音频看起来像卡住。
    """
    script = os.path.join(_HERE, "faithful_transcribe.py")
    tmp = os.path.join(work_dir, "words.json")
    _report(progress, 0.08, f"Loading CrisperWhisper ({model_size}, verbatim)…")
    _dbg("faithful transcribe (streaming) via global py-3.14, model=%s" % model_size)

    proc = subprocess.Popen(
        ["py", "-3.14", script, audio_path, tmp, model_size, "30", "27", str(language)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    # 进度锚点（占总进度 0.08→0.70，后面留给 LLM 和拼接）
    f0, f1, span = 0.08, 0.15, 0.55
    errors: List[str] = []
    tail = collections.deque(maxlen=5)  # 进程意外崩溃时（没有 ERR 行）用来说明原因

    def _reader():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            _dbg("[transcribe] " + line)
            tail.append(line)
            if line.startswith("STAGE LOADING_AUDIO"):
                _report(progress, f0 + 0.03, "Extracting audio (large videos take a few minutes)…")
            elif line.startswith("STAGE LOADING"):
                _report(progress, f0, f"Loading CrisperWhisper ({model_size}, verbatim)…")
            elif line.startswith("STAGE LOADED"):
                _report(progress, f0 + 0.02, "Model loaded, ready to transcribe…")
            elif line.startswith("STAGE AUDIO"):
                _report(progress, f0 + 0.04, "Audio parsed, starting chunked transcription…")
            elif line.startswith("PROGRESS"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        n, ntot = int(parts[1]), int(parts[2])
                        frac = f1 + span * (n / max(1, ntot))
                        _report(progress, min(0.70, frac), f"Transcribing chunks… {n}/{ntot}")
                    except ValueError:
                        pass
            elif line.startswith("ERR "):
                errors.append(line[4:])

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(timeout=2.0)
    if rc != 0:
        detail = errors[-1] if errors else " | ".join(tail)
        raise RuntimeError("CrisperWhisper failed (exit %d): %s" % (rc, detail[-500:]))
    try:
        with open(tmp, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError("CrisperWhisper produced no valid words: %r" % (str(e)[:200]))
    if not data:
        raise RuntimeError("CrisperWhisper returned no words.")
    tokens = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in data]
    _report(progress, 0.70, f"Verbatim transcription complete: {len(tokens)} words")
    return tokens


def _as_posix_clean(word: str) -> str:
    """归一化成可比较的键：保留字母/数字/下划线/撇号以及中日韩等 Unicode 文字，
    去掉空格和标点。中文（无空格分词）也按字符保留，避免被清空导致误判。"""
    if not word:
        return ""
    return re.sub(r"[^\w']", "", word.lower())


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    # 过滤数值异常
    ivs = sorted((s, e) for s, e in intervals if e > s)
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def speech_blocks(tokens: List[Dict], keep_gap: float = 0.8, pad: float = 0.06) -> List[Tuple[float, float]]:
    """把词级时间戳聚成说话片段：[从句]，片段内保留自然停顿，片段间>keep_gap 的静音被切掉。"""
    if not tokens:
        return []
    ts = sorted(tokens, key=lambda t: t["start"])
    blocks: List[Tuple[float, float]] = []
    cur_s, cur_e = ts[0]["start"], ts[0]["end"]
    for t in ts[1:]:
        if t["start"] - cur_e <= keep_gap:
            cur_e = max(cur_e, t["end"])
        else:
            blocks.append((cur_s, cur_e))
            cur_s, cur_e = t["start"], t["end"]
    blocks.append((cur_s, cur_e))
    # 加一点边缘 padding，避免切掉字头字尾
    padded = [(max(0.0, s - pad), e + pad) for s, e in blocks]
    return _merge_intervals(padded)


def _ffprobe(path: str) -> Dict:
    """用 ffprobe 读取音频流参数（codec/sample_rate/channels/bit_rate）及文件时长（秒，读不到时为 None）。"""
    if shutil.which("ffprobe") is None:
        return {}
    try:
        out = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate,bits_per_sample,bits_per_raw_sample",
             "-of", "json", path]))
        s = (out.get("streams") or [{}])[0]
        meta = {"codec_name": s.get("codec_name", "").lower(),
                "sample_rate": s.get("sample_rate") or "44100",
                "channels": s.get("channels") or "2",
                "bit_rate": s.get("bit_rate") or "",
                "bits_per_sample": s.get("bits_per_sample") or 0,
                "bits_per_raw_sample": s.get("bits_per_raw_sample") or 0}
    except Exception:
        meta = {"codec_name": "", "sample_rate": "44100", "channels": "2", "bit_rate": "",
                "bits_per_sample": 0, "bits_per_raw_sample": 0}
    try:
        fmt = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path]))
        meta["duration"] = float(fmt["format"]["duration"])
    except Exception:
        meta["duration"] = None
    return meta


def _audio_duration(path: str) -> float:
    """优先用 ffprobe 的时长；容器里没有时长时流式解码并计数，不把音频读进内存。"""
    d = _ffprobe(path).get("duration")
    if d:
        return d
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = 0
    while True:
        buf = proc.stdout.read(_PIPE_CHUNK)
        if not buf:
            break
        size += len(buf)
    if proc.wait() != 0 or not size:
        raise RuntimeError("Cannot read an audio stream from: " + path)
    return size / 16000.0


def _resolve_output(out_format: str, src_codec: str, src_bit: str, user_bit: str):
    """根据输出选择/源格式，决定 (容器, 编码器, 码率, 是否无损)。"""
    if out_format == "wav":
        return "wav", "pcm_s16le", None, True
    if out_format == "flac":
        return "flac", "flac", None, True
    if out_format == "mp3":
        return "mp3", "libmp3lame", user_bit or "256k", False
    # 自动匹配源
    if src_codec in ("flac", "alac"):
        return "flac", "flac", None, True
    if src_codec in ("pcm_s16le", "pcm_s24le", "pcm_s32le", "wav"):
        return "wav", "pcm_s16le", None, True
    if src_codec == "mp3":
        return "mp3", "libmp3lame", src_bit or user_bit or "256k", False
    if src_codec in ("aac", "m4a"):
        return "mp4", "aac", src_bit or user_bit or "256k", False
    return "wav", "pcm_s16le", None, True


_EXT_BY_FORMAT = {"wav": ".wav", "flac": ".flac", "mp3": ".mp3", "mp4": ".m4a"}


def output_extension(audio_path: str, out_format: str) -> str:
    """输出文件的真实扩展名（out_format 为 auto 时由音频源决定）。"""
    meta = _ffprobe(audio_path)
    fmt = _resolve_output(out_format, meta.get("codec_name", ""), meta.get("bit_rate", "") or "", "")[0]
    return _EXT_BY_FORMAT[fmt]


def _pcm_bits(meta: Dict) -> int:
    """解码用的位深：无损源保持原位深，有损源（ffprobe 报 0）用 16 位。"""
    bits = 0
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        try:
            bits = max(bits, int(meta.get(key) or 0))
        except (TypeError, ValueError):
            pass
    return 16 if bits <= 16 else 24 if bits <= 24 else 32


def _splice_keep(audio_path: str, keep_blocks: List[Tuple[float, float]], out_path: str,
                 out_format: str = "auto", out_bitrate: str = "320k", *,
                 work_dir: str, progress=None) -> None:
    """流式拼接：ffmpeg 解码到管道，只把保留区间写给另一个 ffmpeg 编码。

    全程只在内存里保留一个 _PIPE_CHUNK，不会把整条音轨读进内存，长视频也一样；
    保留原始采样率/声道。先写到 work_dir，成功后才移到 out_path。
    """
    meta = _ffprobe(audio_path)
    fmt, codec, bitrate, _ = _resolve_output(
        out_format, meta.get("codec_name", ""), meta.get("bit_rate", "") or "", out_bitrate)
    sr = int(meta.get("sample_rate") or 44100)
    ch = int(meta.get("channels") or 2)
    bits = _pcm_bits(meta)
    if fmt == "flac":
        bits = min(bits, 24)  # ffmpeg 的 flac 编码器不支持 32 位
    pcm = f"s{bits}le"
    if fmt == "wav":
        codec = f"pcm_{pcm}"
    frame_size = ch * bits // 8

    total_dur = meta.get("duration") or float("inf")
    spans = []
    for s, e in sorted(keep_blocks):
        s_ms = int(max(0.0, min(float(s), total_dur)) * 1000)
        e_ms = int(max(0.0, min(float(e), total_dur)) * 1000)
        if e_ms - s_ms > 40:
            spans.append((int(s_ms * (sr / 1000.0)), int(e_ms * (sr / 1000.0))))
    if not spans:
        raise RuntimeError("No speech content found.")
    _dbg(f"splice: blocks={len(spans)}, audio_dur={total_dur:.2f}s, blocks(first40)={[(round(s,2),round(e,2)) for s,e in keep_blocks][:40]}")

    part = os.path.join(work_dir, "splice.part")
    dec_log_path = os.path.join(work_dir, "splice_decode.log")
    enc_log_path = os.path.join(work_dir, "splice_encode.log")

    def _log_tail(path: str) -> str:
        with open(path, encoding="utf-8", errors="replace") as lf:
            return lf.read()[-300:]

    with open(dec_log_path, "wb") as dec_log, open(enc_log_path, "wb") as enc_log:
        dec = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", audio_path, "-map", "0:a:0", "-vn",
             "-f", pcm, "-acodec", f"pcm_{pcm}", "-ar", str(sr), "-ac", str(ch), "-"],
            stdout=subprocess.PIPE, stderr=dec_log)
        enc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", pcm, "-ar", str(sr), "-ac", str(ch), "-i", "-",
             "-c:a", codec] + (["-b:a", str(bitrate)] if bitrate else []) + ["-f", fmt, part],
            stdin=subprocess.PIPE, stderr=enc_log)

        def _pump(size: int, sink) -> int:
            """从解码管道读 size 字节，写给 sink；sink 为 None 时丢弃。返回实际读到的字节数。"""
            moved = 0
            while moved < size:
                buf = dec.stdout.read(min(_PIPE_CHUNK, size - moved))
                if not buf:
                    break
                if sink is not None:
                    sink.write(buf)
                moved += len(buf)
            return moved

        written = 0
        pos = 0
        last_pct = -1
        end_frame = max(e for _, e in spans)
        try:
            for start, end in spans:
                start = max(start, pos)
                if end <= start:
                    continue
                want_skip = (start - pos) * frame_size
                want_keep = (end - start) * frame_size
                got_skip = _pump(want_skip, None)
                got_keep = _pump(want_keep, enc.stdin) if got_skip == want_skip else 0
                written += got_keep
                pos = end
                if got_skip < want_skip or got_keep < want_keep:  # 音频流比区间短
                    if dec.wait() != 0:
                        raise RuntimeError("ffmpeg decode failed: " + _log_tail(dec_log_path))
                    break
                pct = int(100 * pos / end_frame)
                if pct != last_pct:
                    last_pct = pct
                    _report(progress, 0.90 + 0.05 * pct / 100, f"Splicing… {pct}%")
            enc.stdin.close()
            enc_rc = enc.wait()
        except OSError:  # 编码器提前退出（Windows 上写断开的管道可能报 EINVAL 而不是 EPIPE）
            enc.kill()
            enc_rc = enc.wait() or -1
        finally:
            dec.kill()
            dec.wait()
            dec.stdout.close()
            if enc.poll() is None:
                enc.kill()
                enc.wait()
            try:
                enc.stdin.close()
            except OSError:  # 编码器已退出，缓冲区里剩下的数据写不出去
                pass
    if enc_rc != 0:
        raise RuntimeError("ffmpeg encode failed: " + _log_tail(enc_log_path))
    if not written:
        raise RuntimeError("No audio could be read for the kept segments: " + _log_tail(dec_log_path))
    shutil.move(part, out_path)
    out_dur = _ffprobe(out_path).get("duration")
    _dbg(f"output duration = {out_dur} s")
    print(f"[ok] saved clean audio: {out_path} ({sr}Hz, {ch}ch, {fmt} {bitrate or 'lossless'})")


DEDUP_RULES = (
    "You are an audio editor removing mistakes from a spoken-word recording. "
    "The transcript is shown as utterances (U1, U2, ...); below each utterance every word is on its own line, "
    "formatted as 'GLOBAL_INDEX  WORD'. Decide which words are MIS-SPEAKS that must be deleted, and return their "
    "GLOBAL_INDEX numbers.\n\n"
    "HOW TO JUDGE WHAT IS A MIS-SPEAK (analyze word-by-word which are mistakes/repeats; keep only the last version):\n"
    "CORE RULE — KEEP LAST, DELETE EARLIER: when the SAME phrase or sentence is spoken more than "
    "once — usually because the speaker restarted, re-said it, or corrected — KEEP the LAST occurrence, which is "
    "normally the complete, correct, non-mistake version. DELETE every EARLIER occurrence (the cut-off, shorter, or "
    "broken ones).\n"
    "CLEAR SIGNS OF A MIS-SPEAK / RESTART:\n"
    "  * A word ending in '-' (e.g. 'I-', 'inspired-') = the speaker was cut off mid-word and started over → that "
    "broken cut-off word and the half-finished phrase before the restart are mistakes; delete them.\n"
    "  * The same phrase (e.g. 'inspired by New York City') repeated back-to-back, or with tiny differences → a "
    "restart/stutter; keep only the LAST full version and delete the earlier ones.\n"
    "  * A short fragment that is NOT a complete, coherent idea → a false start; delete it.\n"
    "ALWAYS DELETE (no meaning):\n"
    "  * Interjections / vocal sounds: um, uh, ah, er, erm, mm, hmm, like, you know, i mean, actually, basically, "
    "literally, and bracketed sounds such as [throatclearing], [UM], [UH], [cough], [laughs], [sighs].\n"
    "  * A word repeated consecutively (the the, and and, that that): delete the earlier repeats, keep one.\n"
    "NEVER DELETE:\n"
    "  * A complete, grammatical sentence that carries NEW information — even when it reuses common words or re-mentions "
    "the same topic as a deleted fragment.\n\n"
    "Be CONSERVATIVE: only delete words you are confident are mis-speaks. When in doubt, KEEP. "
    "Return ONLY a JSON object listing the GLOBAL_INDEX numbers to DELETE, e.g. {\"indices\":[0,1,2,5,10,11,12,15]}. "
    "If nothing to delete, return {\"indices\":[]}. JSON only — no explanation."
)


# 系统提示：让模型直接、简洁地返回 JSON，避免长篇推理/解释浪费 token
DEDUP_SYSTEM = (
    "You are a precise audio-editing assistant. You receive a transcript of spoken words, each on its own line as "
    "'GLOBAL_INDEX  WORD'. Your ONLY job is to decide which words are MIS-SPEAKS and must be DELETED, then return them "
    "as a compact JSON object {\"indices\":[...]}. Work directly and tersely: do NOT reason step-by-step, do NOT explain, do NOT "
    "quote or repeat the transcript, and do NOT output anything except the final JSON. Reply immediately with the JSON "
    "only — no thinking block, no commentary."
)

# 中文判重规则（用于中文音频）
DEDUP_RULES_ZH = (
    "你是一名音频编辑，正在清理一段中文口播录音的错漏。转写以话语块（U1, U2, ...）展示，每个词/字单独一行，"
    "格式为『GLOBAL_INDEX  词』。请判断哪些是必须删掉的口误/重复/无意义词，并返回它们的 GLOBAL_INDEX 编号。\n\n"
    "如何判定『口误』（逐字分析哪些是口误/重复，只保留最后一版）：\n"
    "核心规则——删前保后：同一句话或短语被说了不止一次（通常是说话人重新开始、重说或改口）时，保留最后一次"
    "（通常是完整、正确、无误的版本），删掉所有更早的（被截断、更短或残缺的版本）。\n"
    "明确的错误/重开标志：\n"
    "  * 以破折号结尾的词（如『我—』『说—』）= 说话人话说一半被截断又重来 → 这个被截断的词与重开前的半句都是口误，删除。\n"
    "  * 同一短语紧密重复，或只有细微差别（如『今天天气很好 今天天气很好』）→ 重说/口吃；只保留最后一版完整的，删掉更早的。\n"
    "  * 不是完整、连贯句子的短片段 → 开场白/改口；删除。\n"
    "总是删除（无意义）：\n"
    "  * 感叹/语气词与纯声音：嗯、呃、啊、哦、噢、呀、哎、哎呀、哦哦、嗯嗯、哈哈 等，以及括号标注的声音如【咳嗽】【笑】【清嗓子】。\n"
    "  * 连续重复的词（的的、我我、就是说就是说）：删掉更早的重复，保留一个。\n"
    "绝不删除：\n"
    "  * 完整、有语法、携带新信息的句子——即使它复用了常见词，或与某个已删片段提到同一话题。\n\n"
    "要保守：只删你确信是口误的词；拿不准就保留。"
    "只返回一个 JSON 对象，列出要删除的 GLOBAL_INDEX 编号，例如 {\"indices\":[0,1,2,5,10,11,12,15]}。"
    "没有要删的就返回 {\"indices\":[]}。只输出 JSON，不要解释。"
)

# 系统提示：让模型直接、简洁地返回 JSON（中文）
DEDUP_SYSTEM_ZH = (
    "你是一名精确的音频编辑助手。你会收到一段口播逐词转写，每个词/字单独一行，格式为『GLOBAL_INDEX  词』。"
    "你唯一的任务是判断哪些词是『口误』必须删除，然后用一个紧凑的 JSON 对象 {\"indices\":[...]} 返回它们的编号。"
    "直接、简洁地工作：不要一步步推理，不要解释，不要引用或复述转写，除了最后的 JSON 什么都不要输出。"
    "立刻用 JSON 回复——不要思考块，不要任何评论。"
)


# 每个窗口的三段标记：前后上下文只帮助理解语境，只有中间一段的判定会被采用
_WINDOW_NOTES = {
    "en": ("=== CONTEXT BEFORE (read only: judged in another window, may start mid-sentence; do not return these indices) ===",
           "=== WORDS TO JUDGE (return indices from this section only) ===",
           "=== CONTEXT AFTER (read only: shows whether a phrase above is said again, may end mid-sentence; do not return these indices) ==="),
    "zh": ("=== 前文上下文（只读：已在其他窗口判定，开头可能是半句；不要返回这里的编号）===",
           "=== 待判定内容（只返回这一段里的编号）===",
           "=== 后文上下文（只读：用来判断上面的短语后面是否被重说，结尾可能是半句；不要返回这里的编号）==="),
}


def _dedup_prompts(language: str):
    """按语言返回 (规则, 系统提示) 两段判重 prompt。"""
    lang = str(language or "").strip().lower()
    if lang in _ZH_LANGS:
        return DEDUP_RULES_ZH, DEDUP_SYSTEM_ZH
    return DEDUP_RULES, DEDUP_SYSTEM


def _win_lines(words: List[Dict], a: int, b: int) -> List[str]:
    """把 words[a:b] 格式化为按话语分组的每词一行（左为全局索引、右为词）。"""
    sub = words[a:b]
    blocks, cur, cur_idx = [], [], []
    for i, w in enumerate(sub):
        gi = a + i
        cur.append(w["word"]); cur_idx.append(gi)
        if w["word"].rstrip().endswith((".", "!", "?", "…", "。", "！", "？")):
            blocks.append((cur, cur_idx)); cur, cur_idx = [], []
    if cur:
        blocks.append((cur, cur_idx))
    # 过长话语按逗号/长度再拆，让重复短语更显眼
    fin = []
    for ws, idxs in blocks:
        if len(ws) <= 45:
            fin.append((ws, idxs)); continue
        sub2, subi = [], []
        for g, w in zip(idxs, ws):
            sub2.append(w); subi.append(g)
            if len(sub2) >= 45 or w.rstrip().endswith((",", "；", ";")):
                fin.append((sub2, subi)); sub2, subi = [], []
        if sub2:
            fin.append((sub2, subi))
    lines = []
    for bi, (ws, idxs) in enumerate(fin, 1):
        s = words[idxs[0]]["start"]; e = words[idxs[-1]]["end"]
        lines.append("[U%d  %.2f-%.2fs]" % (bi, s, e))
        for g, w in zip(idxs, ws):
            lines.append("  %d  %s" % (g, w))
    return lines


def _resp_text(resp) -> str:
    """从 Responses API 响应里提取文本（output_text 优先，其次从 output 逐项拼）。"""
    txt = getattr(resp, "output_text", None)
    if txt:
        return str(txt)
    parts = []
    for item in (getattr(resp, "output", None) or []):
        for p in (getattr(item, "content", None) or []):
            if getattr(p, "type", None) == "output_text":
                parts.append(getattr(p, "text", "") or "")
    return "".join(parts)


def _strip_json_fences(content: str) -> str:
    """去掉模型包裹 JSON 的 ``` 代码围栏 / 前后说明文字，便于 json.loads。"""
    s = (content or "").strip()
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    end = max(s.rfind("}"), s.rfind("]"))
    if starts and end > min(starts):
        s = s[min(starts):end + 1]
    return s


def _parse_indices(content: str) -> List[int]:
    """从模型回复里解析词索引。回复不是预期结构时抛 ValueError，与「没有要删的」区分开。"""
    parsed = json.loads(_strip_json_fences(content))
    if isinstance(parsed, dict):
        lists = [v for v in parsed.values() if isinstance(v, list)]
        parsed = parsed["indices"] if "indices" in parsed else (lists[0] if len(lists) == 1 else None)
    if not isinstance(parsed, list):
        raise ValueError("unexpected reply shape: %r" % content[:100])
    indices = []
    for x in parsed:
        if isinstance(x, bool):
            continue
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        elif isinstance(x, str) and x.strip().isdigit():
            x = int(x)
        if isinstance(x, int):
            indices.append(x)
    return indices


# 思考强度。DeepSeek V4 系列可选 none（关闭）/ low / high / max，不传时默认开启且为 high。
# 关思考(none)会导致模型逐个词乱删（碎片化、choppy）。判定口误要对照上下文，质量优先，故用 high
REASONING_EFFORT = "high"

# 这些 HTTP 状态表示换一组参数可能成功（提供方不接受某个参数）
_HTTP_BAD_PARAMS = {400, 415, 422}
# 这些 HTTP 状态表示该端点不存在，换另一种 API
_HTTP_NO_ENDPOINT = {404, 405, 501}


def _llm_option_ladder(mode: str) -> List[Dict]:
    """某种 API 可选参数的组合，由全到简。JSON 模式优先，都不行时靠回复文本解析 JSON。"""
    if mode == "chat":
        fmt = {"response_format": {"type": "json_object"}}
        tunings = [{"temperature": 0.0, "reasoning_effort": REASONING_EFFORT},
                   {"reasoning_effort": REASONING_EFFORT}, {"temperature": 0.0}, {}]
    else:
        fmt = {"text": {"format": {"type": "json_object"}}}
        tunings = [{"reasoning": {"effort": REASONING_EFFORT}}, {}]
    return [{**t, **fmt} for t in tunings] + tunings


class _LLMClient:
    """OpenAI 兼容端点的客户端，记住第一组可用的 API 和参数，后续窗口不再重复试错。

    api_mode：
      "chat"       -> 只用 chat.completions
      "responses"  -> 只用 Responses API
      "auto"       -> 先试 Responses API，端点不存在或拒绝请求时回退 chat.completions

    提供方拒绝某个参数（如模型不支持 temperature 或 reasoning）时逐级去掉参数重试。
    鉴权失败、限流、超时、断网时重试参数没有意义，直接抛出。
    """

    def __init__(self, api_url: str, model: str, api_key: str, api_mode: str = "auto", timeout: int = 600):
        from openai import OpenAI
        self._client = OpenAI(base_url=api_url, api_key=api_key, timeout=timeout, max_retries=1)
        self._model = model
        self._modes = {"chat": ["chat"], "responses": ["responses"]}.get(api_mode, ["responses", "chat"])
        self._working = None  # 第一组成功的 (mode, options)

    def _send(self, mode: str, options: Dict, messages: List[Dict], timeout) -> str:
        extra = {"timeout": timeout} if timeout else {}
        if mode == "chat":
            resp = self._client.chat.completions.create(model=self._model, messages=messages, **options, **extra)
            return (getattr(resp.choices[0].message, "content", "") or "").strip()
        resp = self._client.responses.create(model=self._model, input=messages, **options, **extra)
        return _resp_text(resp).strip()

    def complete(self, messages: List[Dict], timeout=None) -> str:
        """返回助手回复的文本；所有组合都失败时抛 RuntimeError，说明最后一个错误。"""
        attempts = [(mode, opts) for mode in self._modes for opts in _llm_option_ladder(mode)]
        if self._working:
            attempts.insert(0, self._working)
        last_error = "empty reply"
        no_endpoint = set()
        for mode, options in attempts:
            if mode in no_endpoint:
                continue
            try:
                txt = self._send(mode, options, messages, timeout)
            except Exception as e:
                status = getattr(e, "status_code", None)
                last_error = str(e)[:300]
                _dbg("[warn] llm %s %s failed: %s" % (mode, sorted(options), last_error))
                if status in _HTTP_NO_ENDPOINT:
                    no_endpoint.add(mode)
                elif status not in _HTTP_BAD_PARAMS:
                    raise RuntimeError(last_error)
                continue
            if txt:
                self._working = (mode, options)
                return txt
            last_error = "empty reply"  # JSON 模式下回复可能只剩空格，去掉参数再试
        raise RuntimeError(last_error)

    def check(self) -> None:
        """发一个极小的请求确认 URL、密钥和模型名可用。"""
        self.complete([{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}], timeout=120)


_LLM_WINDOW_WORDS, _LLM_CONTEXT_WORDS, _LLM_WORKERS = 500, 70, 10


def llm_dedup_words(words: List[Dict], llm: _LLMClient, progress=None,
                    language: str = "en") -> Tuple[List[int], List[str], int]:
    """用 LLM 分窗判定口误，返回 (要删除的全局词索引, 失败窗口的说明, 窗口总数)。

    每窗判定 _LLM_WINDOW_WORDS 个词，前后各带 _LLM_CONTEXT_WORDS 个词的只读上下文。
    规则是「保留最后一次、删除更早的」，所以必须看到后文，才能判定窗口末尾的词后面是否被重说；
    只采用待判定段内的索引，上下文两端的半句话就不会被误判成残句。多窗并行。
    """
    n = len(words)
    if n == 0:
        return [], [], 0
    rules, system_prompt = _dedup_prompts(language)
    before_note, judge_note, after_note = _WINDOW_NOTES[_norm_language(language)]
    windows = [(max(0, a - _LLM_CONTEXT_WORDS), a, min(n, a + _LLM_WINDOW_WORDS),
                min(n, a + _LLM_WINDOW_WORDS + _LLM_CONTEXT_WORDS))
               for a in range(0, n, _LLM_WINDOW_WORDS)]
    total = len(windows)

    def _judge(win) -> Set[int]:
        ctx_start, acpt, acpt_end, ctx_end = win
        lines = [judge_note] + _win_lines(words, acpt, acpt_end)
        if ctx_start < acpt:
            lines = [before_note] + _win_lines(words, ctx_start, acpt) + lines
        if acpt_end < ctx_end:
            lines += [after_note] + _win_lines(words, acpt_end, ctx_end)
        _dbg("==== window (accepted %d-%d) ====\n%s" % (acpt, acpt_end, "\n".join(lines[:25])))
        user_prompt = (rules + "\n\n---\n\nTRANSCRIPT (each word on its own line; the LEFT number is that "
                       "word's GLOBAL index):\n" + "\n".join(lines) +
                       "\n\nReturn ONLY a JSON object {\"indices\":[...]} listing the GLOBAL index numbers to DELETE.")
        content = llm.complete([{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}])
        _dbg("window %d-%d output len=%d" % (acpt, acpt_end, len(content)))
        got = {i for i in _parse_indices(content) if acpt <= i < acpt_end}
        _dbg("window %d-%d -> added %d" % (acpt, acpt_end, len(got)))
        return got

    def _run(win):
        try:
            return _judge(win), None
        except Exception as e:
            _dbg("[warn] window %d-%d failed: %r" % (win[1], win[2], str(e)[:300]))
            return set(), "words %d-%d: %s" % (win[1], win[2], str(e)[:200])

    deleted: Set[int] = set()
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=min(_LLM_WORKERS, total)) as ex:
        futures = [ex.submit(_run, w) for w in windows]
        for done, fut in enumerate(as_completed(futures), 1):
            got, error = fut.result()
            deleted |= got
            if error:
                failures.append(error)
            _report(progress, 0.72 + 0.18 * done / total, "LLM dedup %d/%d windows…" % (done, total))
    _dbg("LLM dedup total deleted = %d, failed windows = %d/%d" % (len(deleted), len(failures), total))
    return sorted(deleted), failures, total


# ---- 用真实音频校准切点 ----
# 词时间戳会把词框窄：实测词尾平均比声音结束早约 90 ms、词头平均晚约 80 ms，个别差几百毫秒。
# 直接切在时间戳上，就会切掉保留词的尾音、带进被删词的尾音，所以要在时间戳附近找真实的声音空隙。
_ENV_RATE = 200           # 响度包络每秒帧数（5 ms 一帧）
_ENV_WINDOW = 160         # 16 kHz 下的 10 ms 窗
_ENV_HOP = 80             # 16 kHz 下的 5 ms 步长
_MIN_VALLEY_SEC = 0.04    # 低于门限持续这么久才算空隙，更短的多半是词内的塞音
_MIN_CONTRAST_DB = 12     # 附近最响和最安静至少差这么多，才能靠响度判断空隙
_SEARCH_SEC = 0.35        # 从时间戳往词外最多找这么远
_NEIGHBOR_SLACK_SEC = 0.10  # 可以越过邻词时间戳这么多：时间戳紧贴时，真实空隙常在它后面一点
_VALLEY_REACH_SEC = 0.3   # 量空隙长度时可以超出搜索范围，否则范围边缘的长空隙会被算短
_TAIL_SEC = 0.04          # 声音结束后再留一点，保住余音
_LEAD_SEC = 0.03          # 声音开始前留一点，保住起音


def _loudness_envelope(path: str) -> np.ndarray:
    """流式解码成 16 kHz 单声道，返回每 5 ms 一帧的响度（dB）。内存只和帧数有关：一小时约 3 MB。"""
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    parts = []
    tail = np.zeros(0, dtype=np.float64)
    pending = b""
    while True:
        buf = proc.stdout.read(_PIPE_CHUNK)
        if not buf:
            break
        buf = pending + buf
        usable = len(buf) - len(buf) % 2  # 读取可能停在半个采样上
        pending = buf[usable:]
        samples = np.concatenate((tail, np.frombuffer(buf[:usable], dtype=np.int16).astype(np.float64)))
        frames = (len(samples) - _ENV_WINDOW) // _ENV_HOP + 1
        if frames <= 0:
            tail = samples
            continue
        energy = np.cumsum(np.concatenate(([0.0], samples ** 2)))
        starts = _ENV_HOP * np.arange(frames)
        parts.append(10 * np.log10((energy[starts + _ENV_WINDOW] - energy[starts]) / _ENV_WINDOW + 1.0).astype(np.float32))
        tail = samples[_ENV_HOP * frames:]
    if proc.wait() != 0 or not parts:
        raise RuntimeError("Cannot read an audio stream from: " + path)
    return np.concatenate(parts)


def _env_frame(env: np.ndarray, t: float) -> int:
    return max(0, min(len(env), int(round(t * _ENV_RATE))))


def _valleys(env: np.ndarray, lo: int, hi: int) -> List[Tuple[int, int]]:
    """[lo, hi) 里所有够长的安静段。门限按前后 5 秒的音量定，适应不同音量和底噪。"""
    around = env[max(0, lo - 1000):hi + 1000]
    floor, speech = np.percentile(around, 5), np.percentile(around, 95)
    if speech - floor < _MIN_CONTRAST_DB:  # 响度几乎不变（持续噪声等），分不出哪里是空隙
        return []
    quiet = env[lo:hi] <= max(floor + 8, speech - 28)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.astype(np.int8), [0]))))
    shortest = int(_MIN_VALLEY_SEC * _ENV_RATE)
    return [(lo + a, lo + b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= shortest]


def _sound_end(env: np.ndarray, end: float, next_start: float) -> Optional[float]:
    """保留词真正结束的位置；附近没有空隙（连读）时返回 None。

    词尾时间戳几乎只会偏早，所以只往后找，第一个空隙就是词的结尾。不选最长的空隙：
    它可能在下一个词的第一个音节之后，那样会把被删词的开头带进来。
    """
    lo = _env_frame(env, end)
    hi = _env_frame(env, min(next_start + _NEIGHBOR_SLACK_SEC, end + _SEARCH_SEC))
    if hi <= lo:
        return None
    reach = int(_VALLEY_REACH_SEC * _ENV_RATE)
    valleys = [v for v in _valleys(env, max(0, lo - reach), min(len(env), hi + reach)) if v[1] > lo and v[0] < hi]
    if not valleys:
        return None
    a, b = valleys[0]
    return float(min(max(a, lo) + _TAIL_SEC * _ENV_RATE, b) / _ENV_RATE)


def _sound_start(env: np.ndarray, start: float, prev_end: float) -> Optional[float]:
    """保留词真正开始的位置；附近没有空隙（连读）时返回 None。

    词头时间戳多半偏晚，所以往前找，离词最近的空隙结束处就是词的开头。
    不越过上一个词的词尾时间戳太多，否则会退回到被删词内部的停顿，把被删词的后半段带进来。
    """
    lo = _env_frame(env, max(prev_end - _NEIGHBOR_SLACK_SEC, start - _SEARCH_SEC))
    hi = _env_frame(env, start)
    if hi <= lo:
        return None
    reach = int(_VALLEY_REACH_SEC * _ENV_RATE)
    valleys = [v for v in _valleys(env, max(0, lo - reach), min(len(env), hi + reach)) if v[1] > lo and v[0] < hi]
    if not valleys:
        return None
    a, b = valleys[-1]
    return float(max(b - _LEAD_SEC * _ENV_RATE, a) / _ENV_RATE)


def clean_keep_blocks(words: List[Dict], del_set: Set[int], keep_gap: float = 0.4, pad: float = 0.06,
                      env: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
    """直接从「保留词」构建保留块。

    被删的词会在中途切断块（不跨越删除区合并），保留块内部保留 <=keep_gap 的自然停顿。
    块的两端先按时间戳加 pad，且 pad 不越过相邻被删词的边界。传入 env（响度包络）时，再把两端
    校准到真实的声音空隙上，保住保留词的尾音，也不带进被删词的尾音；找不到空隙的一端保持不变。
    最后合并相接/重叠的块。
    """
    runs: List[Tuple[int, int, float]] = []
    del_spans: List[Tuple[float, float]] = []
    first = last = None
    cur_e = 0.0
    for i, w in enumerate(words):          # words 已是时间顺序
        s, e = float(w["start"]), float(w["end"])
        if i in del_set:
            # 被删的词：立即闭合当前块，让删除区彻底断开
            if first is not None:
                runs.append((first, last, cur_e))
                first = None
            del_spans.append((s, e))
            continue
        if first is None:
            first, last, cur_e = i, i, e
        elif s - cur_e <= keep_gap:
            last, cur_e = i, max(cur_e, e)
        else:
            runs.append((first, last, cur_e))
            first, last, cur_e = i, i, e
    if first is not None:
        runs.append((first, last, cur_e))
    del_spans.sort()

    blocks: List[Dict] = []
    for a, b, e in runs:
        s = float(words[a]["start"])
        if e - s <= 0.02:
            continue
        ps, pe = max(0.0, s - pad), e + pad
        # 左侧最近的被删词（end < s）：pad 起点不低于它的结尾
        i1 = bisect.bisect_left(del_spans, (s, float("-inf"))) - 1
        if i1 >= 0:
            ps = max(ps, del_spans[i1][1])
        # 右侧最近的被删词（紧贴 e 开始，或与保留词尾部重叠）：pad 终点不越过它的开头
        i2 = bisect.bisect_left(del_spans, (e, float("-inf")))
        if i2 > 0 and del_spans[i2 - 1][1] > e:
            i2 -= 1
        if i2 < len(del_spans):
            pe = min(pe, del_spans[i2][0])
        block = {"first": a, "last": b, "stamp_start": ps, "stamp_end": pe, "start": ps, "end": pe}
        if env is not None:
            sound_start = _sound_start(env, s, float(words[a - 1]["end"]) if a > 0 else 0.0)
            sound_end = _sound_end(env, e, float(words[b + 1]["start"]) if b + 1 < len(words) else float("inf"))
            block["start"] = ps if sound_start is None else sound_start
            block["end"] = pe if sound_end is None else sound_end
        blocks.append(block)
    # 很短的被删词两侧的切点可能校准到一起，两块一合并，被删词就留下来了：这一处退回时间戳
    for prev, cur in zip(blocks, blocks[1:]):
        if cur["first"] > prev["last"] + 1 and cur["start"] <= prev["end"]:
            prev["end"], cur["start"] = prev["stamp_end"], cur["stamp_start"]
    padded = [(block["start"], block["end"]) for block in blocks if block["end"] > block["start"]]
    # 合并相接/重叠的相邻块，避免拼接时同一段被重复（叠音/没对齐）
    merged: List[Tuple[float, float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def dedup_audio(audio_path: str, out_wav: str, api_url: str, model: str, api_key: str,
                whisper_model: str = "large", language: str = "en", keep_gap: float = 0.8,
                out_format: str = "auto", out_bitrate: str = "320k",
                progress=None, api_mode: str = "auto", *, work_dir: str) -> Dict:
    """去重：只保留有文字的段落(词所在区域)，再减去语气词和 LLM 判重标记的词；无文字(静音)自然被切掉。

    返回结果里的 warnings 列出没有完全按要求完成的部分（如 LLM 窗口失败），需要展示给用户。
    """
    language = _norm_language(language)
    total_dur = _audio_duration(audio_path)
    warnings: List[str] = []
    _dbg("==== process start ====")
    _dbg("source total duration = %.2f s (ffprobe)" % total_dur)

    if not (api_url and model and api_key):
        raise RuntimeError("LLM dedup needs an API URL, a model name and an API key.")
    # 转写可能要跑很久：先确认 API 配置可用，配错了立刻报错，不白等
    _report(progress, 0.02, "Checking the LLM API…")
    _dbg("LLM endpoint=%s model=%s api_mode=%s effort=%s" % (api_url, model, api_mode, REASONING_EFFORT))
    llm = _LLMClient(api_url, model, api_key, api_mode)
    try:
        llm.check()
    except Exception as e:
        raise RuntimeError("LLM API check failed, please check the URL, model name and key: %s" % str(e)[:300])

    words = transcribe(audio_path, whisper_model, language=language, progress=progress, work_dir=work_dir)
    if not words:
        raise RuntimeError("Transcription returned no words.")
    _dbg("words = %d; first = %.2f-%.2f; last = %.2f-%.2f"
         % (len(words), words[0]["start"], words[0]["end"], words[-1]["start"], words[-1]["end"]))

    # 只有文字段落：词相邻(gap<=keep_gap)合并成块；无文字(静音/空白)区域自然被排除（仅用于日志参考）
    text_blocks = speech_blocks(words, keep_gap=keep_gap)
    _dbg("text blocks (has text) = %d: %s" % (len(text_blocks), [(round(s,2), round(e,2)) for s,e in text_blocks[:30]]))
    _dbg("text total = %.2f s (of %.2f total)" % (sum(e-s for s,e in text_blocks), total_dur))

    # 无意义声音标记 + 纯语气词必然删，兜底 LLM 的漏判
    core = _core_fillers_for(language)
    del_set: Set[int] = {i for i, w in enumerate(words)
                         if _is_bracket_sound(w["word"]) or _as_posix_clean(w["word"]) in core}
    auto_count = len(del_set)
    _report(progress, 0.7, "LLM dedup: analyzing words window by window…")
    llm_indices, failures, total_windows = llm_dedup_words(words, llm, progress=progress, language=language)
    # 全部失败时成片只切了静音，没有意义，不要当成结果交付
    if len(failures) == total_windows:
        raise RuntimeError("LLM dedup failed for all %d windows. First error: %s" % (total_windows, failures[0]))
    del_set |= set(llm_indices)
    if failures:
        warnings.append("LLM dedup failed for %d of %d windows, so mis-speaks there were NOT removed. First error: %s"
                        % (len(failures), total_windows, failures[0]))
    _dbg("LLM word indices(%d)+auto(%d) -> delete ranges = %s"
         % (len(llm_indices), auto_count,
            [(round(words[i]['start'],2), round(words[i]['end'],2)) for i in sorted(del_set)]))

    # 直接从「保留词」构建保留块，切点校准到真实音频的空隙上；keep_gap 用前端设置控制切分强度
    _report(progress, 0.9, "Refining the cut points on the audio…")
    keep = clean_keep_blocks(words, del_set, keep_gap=keep_gap, pad=0.06, env=_loudness_envelope(audio_path))
    _dbg("keep blocks after LLM = %d: %s" % (len(keep), [(round(s,2), round(e,2)) for s,e in keep[:40]]))
    if not keep:
        raise RuntimeError("Nothing left after filtering.")
    _report(progress, 0.90, f"Splicing ({out_format})…")
    _splice_keep(audio_path, keep, out_wav, out_format=out_format, out_bitrate=out_bitrate,
                 work_dir=work_dir, progress=progress)
    _report(progress, 0.96, "Creating the Premiere project (Premiere opens briefly)…")
    project_path = None
    try:
        project_path, project_warning = premiere.export_project(
            audio_path, keep, os.path.splitext(out_wav)[0], work_dir, log=_dbg)
        if project_warning:
            warnings.append(project_warning)
        # 上传的文件在任务结束时删除，工程里的素材要重新链接；本地路径模式不受影响
        if os.path.abspath(audio_path).startswith(os.path.abspath(work_dir) + os.sep):
            warnings.append("The uploaded source is removed after processing, so Premiere will ask you to relink it. "
                            "Use the local file path option to avoid this.")
    except Exception as e:
        _dbg("[warn] premiere export failed: %r" % (str(e)[:200]))
        warnings.append("The Premiere project could not be created: %s" % str(e)[:200])
    _report(progress, 1.0, "Done")
    kept_sec = round(sum(e - s for s, e in keep), 2)
    removed_sec = round(total_dur - kept_sec, 2)
    _dbg("==== result: kept=%.2f  total=%.2f  removed=%.2f ====" % (kept_sec, total_dur, removed_sec))
    return {
        "tokens": len(words),
        "kept_blocks": len(keep),
        "kept_sec": kept_sec,
        "removed_sec": max(0.0, removed_sec),
        "total_sec": round(total_dur, 2),
        "out_wav": out_wav,
        "out_project": project_path,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--api-mode", choices=["auto", "chat", "responses"], default="auto")
    ap.add_argument("--language", default="en")
    ap.add_argument("--whisper", default="large")
    a = ap.parse_args()
    # 输出容器跟随 --out 的扩展名，否则 AAC 源配 b.wav 会写出内容与扩展名不符的文件
    out_ext = os.path.splitext(a.out)[1].lower().lstrip(".")
    job_dir = new_job_dir()
    try:
        info = dedup_audio(a.audio, a.out, a.api_url, a.model, a.api_key, a.whisper,
                           language=a.language, api_mode=a.api_mode,
                           out_format=out_ext if out_ext in ("wav", "flac", "mp3") else "auto",
                           work_dir=job_dir)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    print(json.dumps(info, indent=2))
