"""
音频自动去重引擎：
  1) CrisperWhisper 逐字转写（在独立解释器中运行，带词级时间戳）
  2) 规则清理：括号声音标记、无歧义语气词（精确、无需模型）
  3) LLM（OpenAI 兼容，chat.completions / responses 端点）逐词判定口误，返回要删除的词索引
  4) 按句判定：LLM 通读删掉口误之后剩下的内容，找出整句该剪的行（画外话、后面又重说了一遍的前一版）。
     读两遍，都认定的才剪，其余的留在成片里并做上标记，交给人工确认
  5) 听成片复核：把剪好的结果拼出来重新逐字转写，和计划保留的词对齐，修掉只有听才能发现的问题
     （被删词漏进来的残音、被切断的词、第一次转写漏掉的口误），再由 LLM 读实际听到的内容，恢复误删、补删漏删
  6) ffmpeg 流式解码 → 只保留选中区间 → 流式编码，内存占用与时长无关
  7) 输出视频和音频都剪好的 Premiere 工程（见 premiere.py）、可通读的剪辑清单，以及逐词的判定记录
     （.work\\edits，供 evaluate.py 和人工修正后的工程对照）
"""
import bisect, collections, datetime, difflib, hashlib, itertools, json, os, re, shutil, subprocess, sys, tempfile, threading, time
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


# 必然删除的无意义语气词（无歧义，直接删，复核时 LLM 也不能恢复；其余交给 LLM 判断）。
# oh 不在这里：它也是数字 0 的读法（"two oh one"），按规则删错了没有人能恢复
CORE_FILLERS = {"um", "uh", "erm", "er", "em", "mm", "hmm", "ah", "uhh", "eh"}
# 语气词表里同时也是单位写法的词：前一个词带数字时（"35 mm"）它是单位，不按规则删
_UNIT_FILLERS = {"mm"}
# 中文口癖/感叹语气词。转写可能按单字切分，所以这里只放几乎不会构成其他词的字；
# 哈（哈佛）、喂（喂养）、吓（吓人）、呵（呵护）、呐（呐喊）这类有歧义的字，
# 粤语里表示「不」的 唔（唔好、唔係：删掉意思就反了），
# 以及“那个/就是/然后”这类有语义的连接词，都交给 LLM 按上下文判断
CORE_FILLERS_ZH = {
    "嗯", "呃", "啊", "哦", "噢", "咦", "诶", "欸", "哎", "唉", "喔",
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


def _is_vocal_sound(word: str, prev_word: str, fillers: Set[str]) -> bool:
    """这个词是不是必然删除的声音：括号声音标记，或语气词表（fillers）里的词。prev_word 是它前面那个词。"""
    if _is_bracket_sound(word):
        return True
    key = _as_posix_clean(word)
    if key not in fillers:
        return False
    return not (key in _UNIT_FILLERS and re.search(r"\d", prev_word))


def _report(progress, frac: float, msg: str):
    if progress:
        try:
            progress(frac, msg)
        except Exception:
            pass


# 中间文件和日志都放在 .work，output 里只留最终结果
WORK_DIR = os.path.join(_HERE, ".work")
# 转写结果跨任务保留（不在 job 目录里）：转写最慢，同一文件重跑时不用再等
TRANSCRIPT_DIR = os.path.join(WORK_DIR, "transcripts")
# 每次任务的逐词判定记录，按输出文件名存放：evaluate.py 用它和人工修正后的 Premiere 工程对照
EDIT_DIR = os.path.join(WORK_DIR, "edits")
# LLM 回复按请求内容存放：同一个文件重跑时判定不变，改了后面的步骤才看得出是哪一步造成的差别
LLM_CACHE_DIR = os.path.join(WORK_DIR, "llm_cache")
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


def _is_upload(audio_path: str, work_dir: str) -> bool:
    """上传的文件存在 job 目录里，任务结束就删，路径每次都不同。"""
    return os.path.abspath(audio_path).startswith(os.path.abspath(work_dir) + os.sep)


def _transcript_cache_path(audio_path: str, model_size: str, language: str, script: str) -> str:
    """转写缓存的路径。文件、模型、语言或转写脚本任何一个变了，都不会命中旧结果。"""
    path = os.path.abspath(audio_path)
    stat = os.stat(path)
    with open(script, "rb") as f:
        script_hash = hashlib.sha256(f.read()).hexdigest()
    key = "|".join(map(str, (os.path.normcase(path), stat.st_size, stat.st_mtime_ns, model_size, language, script_hash)))
    return os.path.join(TRANSCRIPT_DIR, hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json")


def _load_words(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in data]


# 转写子进程被系统异常终止（原生崩溃，不是脚本自己报错退出）时，重新启动的次数和等待秒数。
# 最常见的原因是提交内存（物理内存 + 页面文件）用尽：Windows 扩大页面文件期间内存申请会被拒绝，
# PyTorch 的分配器随即访问冲突。等几秒再启动，页面文件通常已经扩好
_CRASH_RETRIES = 1
_CRASH_RETRY_WAIT = 5.0
# large 模型转写时提交内存峰值约 17 GB（实测）。崩溃时可用量低于这个数，就提示是内存不足
_LOW_COMMIT_GB = 18.0
# 被系统终止时的退出码（Windows NTSTATUS）。Ctrl+C（0xC000013A）是用户主动中断，不算崩溃
_NT_STATUS_NAMES = {
    0xC0000005: "access violation",
    0xC0000017: "out of memory",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC000012D: "commit limit reached",
    0xC0000374: "heap corruption",
    0xC0000409: "fatal runtime error",
}
_NT_CONTROL_C_EXIT = 0xC000013A


def _commit_memory_gb() -> Optional[Tuple[float, float]]:
    """Windows 上返回 (可用提交内存, 提交上限)，单位 GB；其他系统或读不到时返回 None。"""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                    "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPageFile / 2 ** 30, status.ullTotalPageFile / 2 ** 30
    except Exception:
        return None


def _crash_name(rc: int) -> Optional[str]:
    """退出码表示进程被系统异常终止时返回可读的名字，否则（正常退出或脚本自己报错）返回 None。"""
    if os.name == "nt":
        code = rc & 0xFFFFFFFF
        if code < 0xC0000000 or code == _NT_CONTROL_C_EXIT:
            return None
        name = _NT_STATUS_NAMES.get(code)
        return "0x%08X%s" % (code, ", " + name if name else "")
    return "signal %d" % -rc if rc < 0 else None


def _crash_message(crash: str, attempts: int, memory: Optional[Tuple[float, float]], model_size: str) -> str:
    """转写子进程崩溃后给用户看的说明：这不是转写内容的问题，并指出最可能的原因。"""
    msg = "CrisperWhisper crashed (%s)%s. This is a crash inside PyTorch/CUDA, not a problem with the audio." % (
        crash, " %d times in a row" % attempts if attempts > 1 else "")
    if memory and memory[0] < _LOW_COMMIT_GB:
        msg += " Windows was low on memory: %.1f GB of %.1f GB (RAM + page file) was free when it started" % memory
        msg += ", and the large model needs about 17 GB." if model_size == "large" else "."
        msg += " Close memory-heavy apps (Premiere, browsers, games) or enlarge the page file, then run again."
    else:
        msg += " The Python stack at the crash is in .work\\debug.log."
    return msg


def _stage_msg(stage: str, msg: str) -> str:
    """进度文字：stage 为空时原样显示，否则作为前缀（如 "ASR review: "），后面的句子改成小写开头。"""
    return stage + msg[0].lower() + msg[1:] if stage else msg


def _run_transcriber(cmd: List[str], model_size: str, progress, span: Tuple[float, float],
                     stage: str) -> Tuple[int, List[str], List[str]]:
    """运行一次转写子进程。返回 (退出码, 脚本报告的 ERR 行, 最后几行输出)。

    子进程分块转写并实时把 STAGE/PROGRESS 打到 stdout，这里用 Popen + 读线程
    解析并转给 UI，避免长音频看起来像卡住。span 是这次转写占总进度的哪一段。
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    # 进度锚点：这一段的前 1/9 留给加载模型和解码，其余按转写块数推进
    lo, hi = span
    width = hi - lo
    chunks_from = lo + width / 9
    errors: List[str] = []
    tail = collections.deque(maxlen=5)  # 脚本没有报 ERR 就退出时用来说明原因

    def _reader():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            _dbg("[transcribe] " + line)
            tail.append(line)
            if line.startswith("STAGE LOADING_AUDIO"):
                _report(progress, lo + 0.05 * width, _stage_msg(stage, "Extracting audio (large videos take a few minutes)…"))
            elif line.startswith("STAGE LOADING"):
                _report(progress, lo, _stage_msg(stage, f"Loading CrisperWhisper ({model_size}, verbatim)…"))
            elif line.startswith("STAGE LOADED"):
                _report(progress, lo + 0.03 * width, _stage_msg(stage, "Model loaded, ready to transcribe…"))
            elif line.startswith("STAGE AUDIO"):
                _report(progress, lo + 0.07 * width, _stage_msg(stage, "Audio parsed, starting chunked transcription…"))
            elif line.startswith("PROGRESS"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        n, ntot = int(parts[1]), int(parts[2])
                        frac = chunks_from + (hi - chunks_from) * (n / max(1, ntot))
                        _report(progress, min(hi, frac), _stage_msg(stage, f"Transcribing chunks… {n}/{ntot}"))
                    except ValueError:
                        pass
            elif line.startswith("ERR "):
                errors.append(line[4:])

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(timeout=2.0)
    return rc, errors, list(tail)


def transcribe(audio_path: str, model_size: str = "large", language: str = "en", progress=None,
               *, work_dir: str, span: Tuple[float, float] = (0.05, 0.45), stage: str = "",
               out_name: str = "words.json") -> List[Dict]:
    """用 CrisperWhisper verbatim 转写（全局 Python 3.14 + torch），返回词级 token 列表。

    本地路径的文件转写过一次就保存下来，重跑时直接复用（work_dir 里的文件，如上传的文件和待复核的成片，不保存）。
    子进程原生崩溃（多为内存不足）时重新启动一次，仍然崩溃才让任务失败。
    span 是这次转写占总进度的哪一段，stage 是进度文字的前缀，out_name 是结果在 work_dir 里的文件名。
    """
    lo, hi = span
    script = os.path.join(_HERE, "faithful_transcribe.py")
    cache = None if _is_upload(audio_path, work_dir) else _transcript_cache_path(audio_path, model_size, language, script)
    if cache and os.path.isfile(cache):
        try:
            tokens = _load_words(cache)
        except Exception as e:
            tokens = []
            _dbg("[warn] ignoring the saved transcript %s: %r" % (cache, str(e)[:200]))
        if tokens:
            _dbg("reusing saved transcript %s (%d words)" % (cache, len(tokens)))
            _report(progress, hi, _stage_msg(stage, f"Reusing the saved transcript: {len(tokens)} words"))
            return tokens
    tmp = os.path.join(work_dir, out_name)
    # faulthandler：原生崩溃时把 Python 调用栈打出来（进 debug.log），否则只有一个退出码
    cmd = ["py", "-3.14", "-X", "faulthandler", script, audio_path, tmp, model_size, "30", "27", str(language)]
    for attempt in itertools.count(1):
        _report(progress, lo, _stage_msg(stage, f"Loading CrisperWhisper ({model_size}, verbatim)…"))
        memory = _commit_memory_gb()
        _dbg("faithful transcribe (streaming) via global py-3.14, model=%s, file=%s%s" % (
            model_size, audio_path, ", commit memory free %.1f of %.1f GB" % memory if memory else ""))
        rc, errors, tail = _run_transcriber(cmd, model_size, progress, span, stage)
        crash = _crash_name(rc) if rc != 0 else None
        if not crash or attempt > _CRASH_RETRIES:
            break
        _dbg("[warn] the transcriber crashed (%s) on attempt %d, starting it again" % (crash, attempt))
        _report(progress, lo, _stage_msg(stage, "The transcriber crashed (%s), starting it again…" % crash))
        time.sleep(_CRASH_RETRY_WAIT)
    if crash:
        raise RuntimeError(_crash_message(crash, attempt, memory, model_size))
    if rc != 0:
        detail = errors[-1] if errors else " | ".join(tail)
        raise RuntimeError("CrisperWhisper failed (exit %d): %s" % (rc, detail[-500:]))
    try:
        tokens = _load_words(tmp)
    except Exception as e:
        raise RuntimeError("CrisperWhisper produced no valid words: %r" % (str(e)[:200]))
    if not tokens:
        raise RuntimeError("CrisperWhisper returned no words.")
    if cache:
        try:
            os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
            shutil.copyfile(tmp, cache + ".part")
            os.replace(cache + ".part", cache)  # 中途断电也不会留下半个文件
        except OSError as e:
            _dbg("[warn] could not save the transcript: %r" % str(e)[:200])
    _report(progress, hi, _stage_msg(stage, f"Verbatim transcription complete: {len(tokens)} words"))
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


def _splice_spans_ms(keep_blocks: List[Tuple[float, float]], total_dur: float) -> List[Tuple[int, int]]:
    """拼接实际写出的区间（毫秒，素材时间）：限制在素材时长内，按时间排序、互不重叠，去掉太短的。

    拼接和成片时间轴（_EditTimeline）都用它，成片里的时间才能准确换算回素材。
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    for s, e in sorted(keep_blocks):
        # 四舍五入而不是截断：128.29 秒乘 1000 得 128289.99…，截断会少 1 毫秒，
        # 听过的切点原样用到下一遍时，每一遍都会再挪 1 毫秒
        s_ms = max(pos, int(round(max(0.0, min(float(s), total_dur)) * 1000)))
        e_ms = int(round(max(0.0, min(float(e), total_dur)) * 1000))
        if e_ms - s_ms > 40:
            spans.append((s_ms, e_ms))
            pos = e_ms
    return spans


class _EditTimeline:
    """素材时间和成片时间互相换算。blocks 是成片里依次出现的素材区间（秒，按时间排序、互不重叠）。"""

    def __init__(self, blocks: List[Tuple[float, float]]):
        self.blocks = blocks
        self._starts = [s for s, _ in blocks]
        # 每块在成片里的起点；最后一项是成片总长。offsets[1:-1] 就是各个接缝的位置
        self.offsets = list(itertools.accumulate((e - s for s, e in blocks), initial=0.0))

    @classmethod
    def of_splice(cls, keep_blocks: List[Tuple[float, float]], total_dur: float) -> "_EditTimeline":
        """按拼接实际写出的区间建时间轴，和成片逐毫秒对应。"""
        return cls([(s / 1000.0, e / 1000.0) for s, e in _splice_spans_ms(keep_blocks, total_dur)])

    def block_of_source(self, t: float) -> int:
        """素材时间落在第几块；在两块之间（被剪掉）时算前一块，在第一块之前时返回 -1。"""
        return bisect.bisect_right(self._starts, t) - 1

    def to_edit(self, t: float) -> float:
        """素材时间 → 成片时间。被剪掉的时间对应它前面那个接缝。"""
        i = self.block_of_source(t)
        return 0.0 if i < 0 else self.offsets[i] + min(t, self.blocks[i][1]) - self.blocks[i][0]

    def block_of_edit(self, t: float) -> int:
        """成片时间落在第几块。"""
        return max(0, min(len(self.blocks) - 1, bisect.bisect_right(self.offsets, t) - 1))

    def to_source(self, t: float, block: int) -> float:
        """成片时间 → 素材时间，结果限制在第 block 块之内。"""
        s, e = self.blocks[block]
        return max(s, min(e, s + t - self.offsets[block]))


def _splice_keep(audio_path: str, keep_blocks: List[Tuple[float, float]], out_path: str,
                 out_format: str = "auto", out_bitrate: str = "320k", *,
                 work_dir: str, progress=None, span: Tuple[float, float] = (0.90, 0.95),
                 label: str = "Splicing") -> None:
    """流式拼接：ffmpeg 解码到管道，只把保留区间写给另一个 ffmpeg 编码。

    全程只在内存里保留一个 _PIPE_CHUNK，不会把整条音轨读进内存，长视频也一样；
    保留原始采样率/声道。先写到 work_dir，成功后才移到 out_path。
    span 和 label 是这次拼接占总进度的哪一段、进度文字怎么称呼它。
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
    spans = [(int(s_ms * (sr / 1000.0)), int(e_ms * (sr / 1000.0)))
             for s_ms, e_ms in _splice_spans_ms(keep_blocks, total_dur)]
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
                    _report(progress, span[0] + (span[1] - span[0]) * pct / 100, f"{label}… {pct}%")
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
    "The transcript is shown as utterances (U1, U2, ...). Each utterance header shows its text, so you can read it "
    "normally; below the header every word is on its own line, formatted as 'GLOBAL_INDEX  WORD'. Decide which words "
    "are MIS-SPEAKS that must be deleted, and return their GLOBAL_INDEX numbers.\n\n"
    "HOW TO JUDGE WHAT IS A MIS-SPEAK (compare nearby sentences carefully; keep only the last version of anything re-said):\n"
    "CORE RULE — KEEP LAST, DELETE EARLIER: when the SAME phrase or sentence is spoken more than "
    "once because the speaker restarted, re-said it, or corrected it, KEEP the LAST occurrence, which is "
    "normally the complete, correct, non-mistake version. DELETE every EARLIER occurrence (the cut-off, shorter, or "
    "broken ones).\n"
    "CLEAR SIGNS OF A MIS-SPEAK / RESTART:\n"
    "  * A word ending in '-' (e.g. 'I-', 'inspired-') = the speaker was cut off mid-word and started over → that "
    "broken cut-off word and the half-finished phrase before the restart are mistakes; delete them.\n"
    "  * The same phrase (e.g. 'inspired by New York City') repeated back-to-back, or with tiny differences → a "
    "restart/stutter; keep only the LAST full version and delete the earlier ones.\n"
    "  * A fragment the speaker abandons and then restarts or replaces with a corrected version → a false start; delete it.\n"
    "ALWAYS DELETE (no meaning):\n"
    "  * Vocal sounds: um, uh, ah, er, erm, mm, hmm, and bracketed sounds such as [throatclearing], [UM], [UH], [cough], "
    "[laughs], [sighs].\n"
    "  * A word repeated consecutively by accident (the the, and and, that that): delete the earlier repeats, keep one.\n"
    "DELETE ONLY WHEN CLEARLY A VERBAL TIC:\n"
    "  * like, you know, i mean, actually, basically, literally. These often carry meaning ('it looks like this', "
    "'what I mean is', 'it actually failed'): keep them unless the sentence means exactly the same without them and "
    "reads better.\n"
    "NEVER DELETE:\n"
    "  * A complete, grammatical sentence that carries NEW information — even when it reuses common words or re-mentions "
    "the same topic as a deleted fragment.\n"
    "  * Deliberate repetition: emphasis ('really, really fast'), lists, parallel wording, or the same step described "
    "for a different thing.\n"
    "  * A short natural sentence that stands on its own ('Okay.', 'Let's have a look.').\n"
    "CHECK BEFORE ANSWERING: read what remains with your deletions removed. It must read as natural, grammatical speech "
    "with nothing missing. Delete whole mistakes, not scattered words; if a deletion leaves a broken sentence, change it.\n\n"
    "Be CONSERVATIVE: only delete words you are confident are mis-speaks. When in doubt, KEEP. "
    "Return ONLY a JSON object listing the GLOBAL_INDEX numbers to DELETE, e.g. {\"indices\":[0,1,2,5,10,11,12,15]}. "
    "If nothing to delete, return {\"indices\":[]}. JSON only — no explanation."
)


# 系统提示：思考不受限（判定口误要对照上下文，见 REASONING_EFFORT），只要求最终回答是 JSON
DEDUP_SYSTEM = (
    "You are a precise audio-editing assistant. You receive a transcript of spoken words, each on its own line as "
    "'GLOBAL_INDEX  WORD'. Your job is to decide which words are MIS-SPEAKS and must be DELETED. Take the time you need "
    "to compare nearby sentences before deciding. Your final answer must be only a compact JSON object "
    "{\"indices\":[...]}, with no explanation and without quoting the transcript."
)

# 中文判重规则（用于中文音频）
DEDUP_RULES_ZH = (
    "你是一名音频编辑，正在清理一段中文口播录音的错漏。转写以话语块（U1, U2, ...）展示：每块的标题行是这句话的文本，方便正常阅读；"
    "标题下面每个词/字单独一行，格式为『GLOBAL_INDEX  词』。请判断哪些是必须删掉的口误/重复/无意义词，并返回它们的 GLOBAL_INDEX 编号。\n\n"
    "如何判定『口误』（仔细对照前后句子；被重说的内容只保留最后一版）：\n"
    "核心规则——删前保后：同一句话或短语因为说话人重新开始、重说或改口而被说了不止一次时，保留最后一次"
    "（通常是完整、正确、无误的版本），删掉所有更早的（被截断、更短或残缺的版本）。\n"
    "明确的错误/重开标志：\n"
    "  * 以破折号结尾的词（如『我—』『说—』）= 说话人话说一半被截断又重来 → 这个被截断的词与重开前的半句都是口误，删除。\n"
    "  * 同一短语紧密重复，或只有细微差别（如『今天天气很好 今天天气很好』）→ 重说/口吃；只保留最后一版完整的，删掉更早的。\n"
    "  * 说了一半就放弃、随后重说或改口的片段 → 开头说错；删除。\n"
    "总是删除（无意义）：\n"
    "  * 感叹/语气词与纯声音：嗯、呃、啊、哦、噢、呀、哎、哎呀、哦哦、嗯嗯、哈哈 等，以及括号标注的声音如【咳嗽】【笑】【清嗓子】。\n"
    "  * 无意中连续重复的词（的的、我我、就是说就是说）：删掉更早的重复，保留一个。\n"
    "只有明显是口头禅才删除：\n"
    "  * 那个、这个、就是、然后、对吧。它们常有实际作用（『那个文件』『然后我们再看』）：除非删掉后意思完全不变、读起来更顺，否则保留。\n"
    "绝不删除：\n"
    "  * 完整、有语法、携带新信息的句子——即使它复用了常见词，或与某个已删片段提到同一话题。\n"
    "  * 有意的重复：强调（『非常非常快』）、排比、列举，或对不同对象说的同样步骤。\n"
    "  * 能独立成句的自然短句（『好的。』『我们来看一下。』）。\n"
    "回答前检查：通读删除后剩下的内容，它必须是自然、通顺、没有缺漏的口语。要整段删除错误，不要东删一个词西删一个词；"
    "如果删除后句子不通，就调整。\n\n"
    "要保守：只删你确信是口误的词；拿不准就保留。"
    "只返回一个 JSON 对象，列出要删除的 GLOBAL_INDEX 编号，例如 {\"indices\":[0,1,2,5,10,11,12,15]}。"
    "没有要删的就返回 {\"indices\":[]}。只输出 JSON，不要解释。"
)

# 系统提示（中文）：思考不受限，只要求最终回答是 JSON
DEDUP_SYSTEM_ZH = (
    "你是一名精确的音频编辑助手。你会收到一段口播逐词转写，每个词/字单独一行，格式为『GLOBAL_INDEX  词』。"
    "你的任务是判断哪些词是『口误』必须删除。做判断前请花足够的时间对照前后句子。"
    "最终回答只能是一个紧凑的 JSON 对象 {\"indices\":[...]}，不要解释，不要引用或复述转写。"
)


# 第二轮复核：换一个角度，不再找口误，而是检查成片。成片先拼出来重新逐字转写（见 _merge_heard），
# 所以这里没标 DEL 的词就是成片里实际听到的内容
REVIEW_RULES = (
    "You are reviewing an edit of a spoken-word recording. Another editor chose which words to delete, "
    "to remove mis-speaks while keeping everything the speaker meant to say. The cut audio was then played back to a "
    "speech recognizer, and this transcript shows what it heard. "
    "The transcript is shown as utterances (U1, U2, ...). Each utterance header shows its text with the deleted words "
    "marked [-like this-]. Below the header every word is on its own line as 'GLOBAL_INDEX  WORD', or "
    "'GLOBAL_INDEX  DEL  WORD' if it is deleted. "
    "The words without a mark are what a listener hears in the cut audio, including words the first transcript missed. "
    "The words marked DEL were cut out and are not heard. "
    "A word marked LEFTOVER ([~like this~] in the header) was heard right at a cut although it was not in the plan: "
    "usually it is a fragment of the cut-out words that leaked through, so it is being trimmed and will not be heard "
    "either. "
    "A word marked OFF ([=like this=] in the header) belongs to a whole line that was cut on purpose, because the "
    "speaker was off-script (talking to themselves, a remark about the recording) or said the same thing again later. "
    "Those cuts are final: never list them.\n\n"
    "Read the text the way a listener will hear it, with the deleted words gone, and look for two kinds of error:\n"
    "RESTORE — a word was wrongly deleted:\n"
    "  * Without it the remaining text loses information or is no longer grammatical.\n"
    "  * It was deleted as a repeat, but what the speaker says later is not really the same thing, so nothing replaces it.\n"
    "  * It is the only complete version: the later version that was kept is cut off or broken. Restore the complete "
    "version and delete the broken one.\n"
    "  * It carries meaning ('it looks like this', 'it actually failed'), or it is deliberate repetition for emphasis, "
    "part of a list, or a short natural sentence that stands on its own.\n"
    "  * A LEFTOVER is really a word of the sentence that the first transcript missed ('Then we connect the two "
    "lines [~as well.~]'). Restore it only when it clearly belongs: a LEFTOVER that repeats or half-repeats the cut-out words "
    "next to it, or that does not fit the sentence, stays trimmed.\n"
    "DELETE — something that should not be heard is still there:\n"
    "  * A cut-off word (ending in '-'), a false start or a re-said phrase is still in the kept text. Keep only the "
    "last full version.\n"
    "  * The join around a cut now repeats a word or leaves part of a phrase behind ('we can we can go').\n"
    "  * One or two kept words are stranded between two cuts and no longer connect to anything.\n\n"
    "Most cuts are right, so only change what is clearly wrong. Vocal sounds such as um, uh and [cough] are "
    "always removed: never list them. "
    "Return ONLY a JSON object, e.g. {\"restore\":[12,13],\"delete\":[40]}: 'restore' lists words marked DEL or "
    "LEFTOVER that must stay, and 'delete' lists unmarked words that must go. If nothing needs to change, return "
    "{\"restore\":[],\"delete\":[]}. JSON only — no explanation."
)

REVIEW_SYSTEM = (
    "You are a careful audio-editing reviewer. You check another editor's cuts against a transcript of the cut audio "
    "and correct only clear errors. Take the time you need to read the result as a listener would. Your final answer "
    "must be only "
    "a compact JSON object {\"restore\":[...],\"delete\":[...]}, with no explanation and without quoting the transcript."
)

REVIEW_RULES_ZH = (
    "你正在复核一段中文口播录音的剪辑成片。另一位编辑已经选好了要删除的词，目标是去掉口误，同时保留说话人想说的所有内容。"
    "剪好的音频随后交给语音识别重新听了一遍，下面的转写就是它听到的结果。"
    "转写以话语块（U1, U2, ...）展示：每块的标题行是这句话的文本，被删除的部分标成 [-这样-]；"
    "标题下面每个词/字单独一行，格式为『GLOBAL_INDEX  词』，被删除的格式为『GLOBAL_INDEX  DEL  词』。"
    "没有标记的词就是听众在成片里听到的内容，其中包括第一次转写漏掉的词。标了 DEL 的词已被剪掉，听不到。"
    "标了 LEFTOVER 的词（标题行里写成 [~这样~]）是在剪切处听到、但不在计划里的声音："
    "它通常是被删词漏进来的残音，所以正在被修掉，同样听不到。"
    "标了 OFF 的词（标题行里写成 [=这样=]）属于被整行剪掉的内容：说话人在说画外话（自言自语、关于录制本身的话），"
    "或者后面又把同样的内容说了一遍。这些删除是有意的、已经定下来的，不要列出。\n\n"
    "请按听众听到的内容阅读（被删的词已经不存在），查找两类错误：\n"
    "RESTORE（恢复）——词被误删：\n"
    "  * 删掉后，剩下的内容丢失了信息，或者句子不通。\n"
    "  * 它被当成重复删掉，但说话人后面说的其实不是同一件事，没有内容可以替代它。\n"
    "  * 它是唯一完整的版本：被保留的后一版反而残缺或被截断。这时恢复完整版，删除残缺版。\n"
    "  * 它有实际作用（『那个文件』『然后我们再看』），或者是有意的强调、排比、列举，或者是能独立成句的自然短句。\n"
    "  * 某个 LEFTOVER 其实是这句话里的词，只是第一次转写漏掉了（『然后把两条线也连起来[~吧~]』）。"
    "只有它明显属于这句话时才恢复：重复或半重复旁边被删内容的、放进句子里不通的，继续修掉。\n"
    "DELETE（补删）——成片里还有不该听到的内容：\n"
    "  * 保留的内容里还有被截断的词（以破折号结尾）、开头说错的片段或重说的短语。只保留最后一版完整的。\n"
    "  * 剪切处前后现在重复了一个词，或者留下了半个短语（『我们可以我们可以看』）。\n"
    "  * 一两个保留的词孤零零夹在两处删除之间，前后都接不上。\n\n"
    "大多数删除是对的，只改明显有错的地方。嗯、呃、【咳嗽】这类声音总是会被删除，不要列出。"
    "只返回一个 JSON 对象，例如 {\"restore\":[12,13],\"delete\":[40]}：restore 列出标了 DEL 或 LEFTOVER 但必须保留的词，"
    "delete 列出没有标记但必须删除的词。没有要改的就返回 {\"restore\":[],\"delete\":[]}。只输出 JSON，不要解释。"
)

REVIEW_SYSTEM_ZH = (
    "你是一名细致的音频剪辑复核员。你对照成片的转写检查另一位编辑的删除，只纠正明显的错误。"
    "请花足够的时间，按听众听到的内容通读结果。"
    "最终回答只能是一个紧凑的 JSON 对象 {\"restore\":[...],\"delete\":[...]}，不要解释，不要引用或复述转写。"
)


# 按句判定：第一轮逐词判定只认「同一句话说了两遍」，认不出画外话和换了说法的重说。对照人工修正过的工程实测：
# 第一轮删的词几乎全对，但该删的只找到约六成，漏掉的九成是整句的画外话和整句重说（见 evaluate.py）。
# 所以这一轮换成编辑的角度：读删掉口误之后剩下的内容，一行一行判断它该不该出现在成片里
TAKE_RULES = (
    "You are editing the narration of a recorded video. The recording is raw: besides the narration it contains "
    "things the speaker never meant the audience to hear. Stutters, false starts and filler sounds have already been "
    "removed. What is left is shown below in the order it was spoken, one numbered line (L0, L1, ...) per sentence or "
    "phrase. A note such as '(after a 12 s pause)' means the speaker was silent for that long before the line.\n\n"
    "Find the lines that must be cut from the finished video. There are two kinds.\n\n"
    "OFF-SCRIPT — the speaker is not narrating to the audience:\n"
    "  * Talking to themselves, or reacting to something that went wrong while recording ('Oh god, wrong button.', "
    "'Why is it so laggy?', 'Nope.', 'Hang on a minute.', 'What am I doing?').\n"
    "  * Remarks about the recording, or notes to themselves or to the editor ('I'll record the voice for this later.', "
    "'Let me try that again.', 'Stop there.', 'Put this earlier.', 'What else is on the list?', 'Is it outro time?').\n"
    "  * Apologies and comments on their own performance ('Sorry, I keep stumbling over my words today.').\n"
    "  * Warm-up talk before the narration begins, and chatter after it ends.\n\n"
    "EARLIER TAKE — the speaker says something and later says it again, usually after a pause, a mistake or an "
    "off-script remark. The wording can be quite different: what matters is that the later line delivers the same "
    "point, so a viewer who hears both hears it twice. Cut the EARLIER take and keep the LAST one. A take can span "
    "several lines: list every line of the earlier take. A line that starts a thought and drops it is also an "
    "earlier take when a later line starts that thought again.\n\n"
    "KEEP:\n"
    "  * Narration that is said once, however short or casual ('Okay.', 'Let's have a look.', 'And there we go.'). "
    "A conversational tone is not off-script when the speaker is talking to the viewer.\n"
    "  * Reactions that are part of the video: the narration carries on from them as if the viewer heard them.\n"
    "  * Lines that come back to a topic to ADD something: a summary, another example, the same step for a different "
    "thing. A line is an earlier take only when a later line makes it redundant.\n"
    "  * The LAST take of anything, even when it is not perfect.\n\n"
    "Only cut a line when you are confident. If you suspect a line but are not sure, put its number in 'unsure' "
    "instead: it stays in the video and is marked for a human editor to check. Most lines are narration and stay.\n"
    "Return ONLY a JSON object, e.g. {\"off_script\":[14,15],\"earlier_take\":[[31,34],[32,34]],\"unsure\":[40]}: "
    "'off_script' lists line numbers, 'earlier_take' lists pairs [line to cut, later line that replaces it], and "
    "'unsure' lists line numbers. If nothing needs to be cut, return "
    "{\"off_script\":[],\"earlier_take\":[],\"unsure\":[]}. JSON only — no explanation."
)

TAKE_SYSTEM = (
    "You are an experienced video editor preparing a raw voice recording for publication. You read the transcript "
    "line by line and decide which whole lines do not belong in the finished video: off-script remarks, and earlier "
    "takes of something the speaker says again later. Take the time you need to compare each line with the lines "
    "after it. Your final answer must be only a compact JSON object "
    "{\"off_script\":[...],\"earlier_take\":[[...]],\"unsure\":[...]}, with no explanation and without quoting the "
    "transcript."
)

TAKE_RULES_ZH = (
    "你正在剪辑一段视频的口播旁白。录音是原始素材：除了旁白，里面还有说话人并不想让观众听到的内容。"
    "口吃、说错重来的半句和语气词已经删掉了。剩下的内容按说话顺序列在下面，每句话或每个短语一行，带编号（L0, L1, ...）。"
    "行前的『（停顿 12 秒后）』表示说这一行之前，说话人沉默了这么久。\n\n"
    "请找出成片里必须剪掉的行。有两类。\n\n"
    "画外话（off_script）——说话人不是在对观众讲解：\n"
    "  * 自言自语，或者对录制中出的状况做出的反应（『哎呀按错了。』『怎么这么卡？』『不对。』『等一下。』『我在干嘛？』）。\n"
    "  * 关于录制本身的话，或者给自己、给剪辑师的备注（『这段配音我后面再录。』『我重来一遍。』『停在这。』『这段放前面。』"
    "『清单上还有什么？』『该说结尾了吗？』）。\n"
    "  * 道歉，或者评价自己的发挥（『不好意思，今天老是嘴瓢。』）。\n"
    "  * 旁白开始之前的热身闲话，旁白结束之后的闲聊。\n\n"
    "前一版（earlier_take）——说话人说了一段内容，后面又说了一遍，中间通常隔着停顿、口误或画外话。"
    "两遍的措辞可能差别很大：关键是后面那一行讲的是同一个意思，两行都留着，观众就会听到两遍。"
    "剪掉前面的版本，保留最后一版。一版可能占好几行：把前一版的每一行都列出来。"
    "一行话开了个头就放弃了，后面又有一行重新开这个头，前面那行也算前一版。\n\n"
    "保留：\n"
    "  * 只说了一遍的旁白，不管多短、多随意（『好。』『我们来看一下。』『这样就行了。』）。"
    "说话人是在对观众说话时，口语化的语气不算画外话。\n"
    "  * 属于视频内容的反应：后面的旁白接着它往下讲，就像观众听到了它一样。\n"
    "  * 回到同一个话题、但补充了新内容的行：总结、另一个例子、对另一个对象做同样的步骤。"
    "只有后面的行让它变得多余时，它才是前一版。\n"
    "  * 任何内容的最后一版，即使它并不完美。\n\n"
    "有把握才剪。怀疑某一行该剪但拿不准，就把行号放进 unsure：这一行会留在成片里，并做上标记交给人工确认。"
    "大多数行是旁白，应当保留。\n"
    "只返回一个 JSON 对象，例如 {\"off_script\":[14,15],\"earlier_take\":[[31,34],[32,34]],\"unsure\":[40]}："
    "off_script 列出行号，earlier_take 列出 [要剪掉的行, 取代它的后面那一行] 这样的数对，unsure 列出拿不准的行号。"
    "没有要剪的就返回 {\"off_script\":[],\"earlier_take\":[],\"unsure\":[]}。只输出 JSON，不要解释。"
)

TAKE_SYSTEM_ZH = (
    "你是一名经验丰富的视频剪辑师，正在把一段原始口播录音整理成可以发布的成片。"
    "你逐行阅读转写，判断哪些整行不该出现在成片里：画外话，以及后面又重说了一遍的前一版。"
    "做判断前请花足够的时间，把每一行和它后面的行对照。"
    "最终回答只能是一个紧凑的 JSON 对象 {\"off_script\":[...],\"earlier_take\":[[...]],\"unsure\":[...]}，"
    "不要解释，不要引用或复述转写。"
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


def _review_prompts(language: str):
    """按语言返回 (规则, 系统提示) 两段复核 prompt。"""
    return (REVIEW_RULES_ZH, REVIEW_SYSTEM_ZH) if _norm_language(language) == "zh" else (REVIEW_RULES, REVIEW_SYSTEM)


# 按句判定的三段标记。后文上下文里的行可以是「取代前一版的后一行」，但它自己该不该剪由别的窗口判定
_LINE_WINDOW_NOTES = {
    "en": ("=== CONTEXT BEFORE (read only: judged in another window; do not cut these lines) ===",
           "=== LINES TO JUDGE (only cut lines from this section) ===",
           "=== CONTEXT AFTER (read only: shows whether a line above is said again, and a line here can be the later "
           "line of a pair; do not cut these lines) ==="),
    "zh": ("=== 前文上下文（只读：已在其他窗口判定；不要剪这里的行）===",
           "=== 待判定的行（只剪这一段里的行）===",
           "=== 后文上下文（只读：用来判断上面的行后面是否又说了一遍，这里的行可以作为数对里取代前一版的后一行；"
           "不要剪这里的行）==="),
}


def _take_prompts(language: str):
    """按语言返回 (规则, 系统提示) 两段按句判定 prompt。"""
    return (TAKE_RULES_ZH, TAKE_SYSTEM_ZH) if _norm_language(language) == "zh" else (TAKE_RULES, TAKE_SYSTEM)


# 句末标点：转写在这些标点后面断句
_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
# 被按句判定整行剪掉的词，cut 记成这两个值之一：画外话 / 后面又说了一遍的前一版
_TAKE_CUTS = ("off", "retake")


def _utterances(words: List[Dict], a: int, b: int) -> List[List[int]]:
    """把 words[a:b] 按句子分组，返回每句的全局词索引。过长的句子按逗号/长度再拆，让重复短语更显眼。"""
    blocks, cur = [], []
    for i in range(a, b):
        cur.append(i)
        if words[i]["word"].rstrip().endswith(_SENTENCE_END):
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    fin = []
    for idxs in blocks:
        if len(idxs) <= 45:
            fin.append(idxs); continue
        part = []
        for i in idxs:
            part.append(i)
            if len(part) >= 45 or words[i]["word"].rstrip().endswith((",", "；", ";")):
                fin.append(part); part = []
        if part:
            fin.append(part)
    return fin


def _word_sep(language: str) -> str:
    """把词拼成句子时用的分隔符：中文逐字转写，不加空格。"""
    return "" if _norm_language(language) == "zh" else " "


def _marked_text(words: List[Dict], idxs: List[int], sep: str, mark) -> str:
    """把一句词拼成文本。mark(i) 返回这个词的标记格式（如 "[-%s-]"），None 表示不标记；相邻同标记的词合在一起。"""
    parts = []
    for fmt, group in itertools.groupby(idxs, key=mark):
        text = sep.join(words[i]["word"].strip() for i in group)
        parts.append(fmt % text if fmt else text)
    return " ".join(parts)


def _win_lines(words: List[Dict], a: int, b: int, sep: str = " ", proposed: Set[int] = frozenset()) -> List[str]:
    """把 words[a:b] 格式化为按话语分组的文本：标题行带整句（方便发现重复短语），下面每词一行（左为全局索引、右为词）。

    proposed 是复核时已经删除的词：标题行里标成 [-这样-]，词行里标 DEL。其中听成片时在剪切处多听到的声音
    （cut 为 "leak"）标成 [~这样~] 和 LEFTOVER：它们多半是被删词的残音，但也可能是第一次转写漏掉的词，要让 LLM 看得出区别。
    按句判定整行剪掉的词（cut 为 _TAKE_CUTS 之一）标成 [=这样=] 和 OFF：它们不是口误，复核不能恢复，也要看得出区别。
    """
    def _flag(i: int) -> Optional[str]:
        if i not in proposed:
            return None
        cut = words[i].get("cut")
        return "LEFTOVER" if cut == "leak" else "OFF" if cut in _TAKE_CUTS else "DEL"

    marks = {"DEL": "[-%s-]", "LEFTOVER": "[~%s~]", "OFF": "[=%s=]", None: None}
    lines = []
    for bi, idxs in enumerate(_utterances(words, a, b), 1):
        text = _marked_text(words, idxs, sep, lambda i: marks[_flag(i)])
        lines.append("[U%d  %.2f-%.2fs]  %s" % (bi, words[idxs[0]]["start"], words[idxs[-1]]["end"], text))
        lines.extend("  %d  %s%s" % (i, _flag(i) + "  " if _flag(i) else "", words[i]["word"]) for i in idxs)
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


def _strip_json_fences(content: str, key: str = "indices") -> str:
    """去掉模型包裹 JSON 的 ``` 代码围栏 / 前后说明文字，便于 json.loads。"""
    # 有些提供方把思考过程直接放在回复文本里，其中的括号会干扰下面找 JSON 的边界
    s = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    # 没开 JSON 模式时模型可能先写推理再给答案，推理里的 [UM] 之类也会干扰，优先取最后一个答案对象
    answers = [m for m in re.findall(r"\{[^{}]*\}", s) if '"%s"' % key in m]
    if answers:
        return answers[-1]
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    end = max(s.rfind("}"), s.rfind("]"))
    if starts and end > min(starts):
        s = s[min(starts):end + 1]
    return s


def _index_list(values) -> List[int]:
    """把回复里的一个列表转成词索引。不是列表时抛 ValueError，与「没有要改的」区分开。"""
    if not isinstance(values, list):
        raise ValueError("unexpected reply shape: %r" % str(values)[:100])
    indices = []
    for x in values:
        if isinstance(x, bool):
            continue
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        elif isinstance(x, str) and x.strip().isdigit():
            x = int(x)
        if isinstance(x, int):
            indices.append(x)
    return indices


def _parse_indices(content: str) -> List[int]:
    """从第一轮的回复里解析要删除的词索引。"""
    parsed = json.loads(_strip_json_fences(content))
    if isinstance(parsed, dict):
        lists = [v for v in parsed.values() if isinstance(v, list)]
        parsed = parsed["indices"] if "indices" in parsed else (lists[0] if len(lists) == 1 else None)
    return _index_list(parsed)


def _parse_review(content: str) -> Tuple[Set[int], Set[int]]:
    """从复核的回复里解析 (要恢复的词, 要补删的词)。"""
    parsed = json.loads(_strip_json_fences(content, "restore"))
    if not (isinstance(parsed, dict) and {"restore", "delete"} & parsed.keys()):
        raise ValueError("unexpected reply shape: %r" % content[:100])
    return set(_index_list(parsed.get("restore", []))), set(_index_list(parsed.get("delete", [])))


def _parse_takes(content: str) -> Tuple[Set[int], List[Tuple[int, int]], Set[int]]:
    """从按句判定的回复里解析 (画外话的行号, [(前一版的行, 取代它的后一行)], 拿不准的行号)。

    前一版必须说出是哪一行取代了它：只给了行号、没给后一行的，没有依据核对，算作拿不准。
    """
    parsed = json.loads(_strip_json_fences(content, "off_script"))
    if not (isinstance(parsed, dict) and {"off_script", "earlier_take"} & parsed.keys()):
        raise ValueError("unexpected reply shape: %r" % content[:100])
    takes = parsed.get("earlier_take", [])
    if not isinstance(takes, list):
        raise ValueError("unexpected reply shape: %r" % str(takes)[:100])
    pairs: List[Tuple[int, int]] = []
    unsure = set(_index_list(parsed.get("unsure", [])))
    for item in takes:
        numbers = _index_list(item if isinstance(item, list) else [item])
        if len(numbers) == 2:
            pairs.append((numbers[0], numbers[1]))
        else:
            unsure.update(numbers[:1])
    return set(_index_list(parsed.get("off_script", []))), pairs, unsure


# 思考强度。DeepSeek V4 系列可选 none（关闭）/ low / high / max，不传时默认开启且为 high。
# 关思考(none)会导致模型逐个词乱删（碎片化、choppy）；实际使用中 high 的结果也比 max 好，所以不用 max
REASONING_EFFORT = "high"

# 这些 HTTP 状态表示换一组参数可能成功（提供方不接受某个参数）
_HTTP_BAD_PARAMS = {400, 415, 422}
# 这些 HTTP 状态表示该端点不存在，换另一种 API
_HTTP_NO_ENDPOINT = {404, 405, 501}
# 这些 HTTP 状态表示密钥或余额有问题，重试窗口也不会成功
_HTTP_NO_RETRY = {401, 402, 403}


def _llm_option_ladder(mode: str) -> List[Dict]:
    """某种 API 可选参数的组合，由全到简。思考强度比 JSON 模式重要（JSON 可以从回复文本里解析），
    所以先去掉 JSON 模式，最后才去掉思考强度。"""
    if mode == "chat":
        fmt = {"response_format": {"type": "json_object"}}
        tunings = [{"temperature": 0.0, "reasoning_effort": REASONING_EFFORT},
                   {"reasoning_effort": REASONING_EFFORT}, {"temperature": 0.0}, {}]
    else:
        fmt = {"text": {"format": {"type": "json_object"}}}
        tunings = [{"reasoning": {"effort": REASONING_EFFORT}}, {}]
    return [options for t in tunings for options in ({**t, **fmt}, t)]


# 参数里要求 JSON 模式的键（chat / responses 各一个）
_LLM_FORMAT_KEYS = ("response_format", "text")


class _LLMClient:
    """OpenAI 兼容端点的客户端，记住第一组可用的 API 和参数，后续窗口都用这一组，不再重复试错。

    api_mode：
      "chat"       -> 只用 chat.completions
      "responses"  -> 只用 Responses API
      "auto"       -> 先试 Responses API，端点不存在或拒绝请求时回退 chat.completions

    第一次请求（check）时，提供方拒绝某个参数（如模型不支持 temperature 或 reasoning）就逐级去掉参数重试，
    找到可用的一组后固定下来（见 _attempts）。
    鉴权失败、限流、超时、断网时重试参数没有意义，直接抛出。
    """

    def __init__(self, api_url: str, model: str, api_key: str, api_mode: str = "auto", timeout: int = 600):
        from openai import OpenAI
        self._client = OpenAI(base_url=api_url, api_key=api_key, timeout=timeout, max_retries=1)
        self._endpoint = api_url
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

    def _attempts(self) -> List[Tuple[str, Dict]]:
        """这次请求依次尝试的 (API 种类, 参数)。

        还没有成功过：所有组合，由全到简。已经成功过：只用那一组，至多去掉 JSON 模式再试一次
        （回复文本里也能解析出 JSON）。不再往下试别的组合：提供方偶发一次 400，就会让这一窗、
        乃至后面所有窗口在不带思考的情况下判定，而且没有任何提示。宁可让这一窗失败重试，重试用尽就如实报告。
        """
        if not self._working:
            return [(mode, opts) for mode in self._modes for opts in _llm_option_ladder(mode)]
        mode, options = self._working
        plain = {k: v for k, v in options.items() if k not in _LLM_FORMAT_KEYS}
        return [(mode, options)] + ([(mode, plain)] if plain != options else [])

    def complete(self, messages: List[Dict], timeout=None) -> str:
        """返回助手回复的文本；所有组合都失败时抛 RuntimeError，说明最后一个错误。"""
        last_error = "empty reply"
        no_endpoint = set()
        for mode, options in self._attempts():
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
                    raise
                continue
            if txt:
                if not self._working:
                    self._working = (mode, options)
                return txt
            last_error = "empty reply"  # JSON 模式下回复可能只剩空格，去掉参数再试
        raise RuntimeError(last_error)

    @property
    def working(self) -> Optional[Tuple[str, Dict]]:
        """第一组成功的 (API 种类, 参数)。"""
        return self._working

    @property
    def cache_id(self) -> str:
        """区分回复来源的标识：接口地址、模型和参数任何一个变了，存下的回复都不再适用。"""
        return json.dumps([self._endpoint, self._model, self._working], sort_keys=True)

    def check(self) -> None:
        """发一个极小的请求确认 URL、密钥和模型名可用。"""
        self.complete([{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}], timeout=120)


def _ask(llm, messages: List[Dict], parse, label: str, ask: int = 0):
    """问 LLM 并用 parse 解析回复，返回解析结果；parse 不接受回复时抛出它的异常（调用方据此重试）。

    解析通过的回复按请求内容存盘（LLM_CACHE_DIR），同样的请求下次直接用存下的回复：同一个文件重跑时判定不变，
    改了后面的步骤才看得出差别是哪一步造成的。请求内容（提示词、转写、判定范围）或 llm.cache_id 变了就不会命中。
    没通过解析的回复不存，所以一次坏的回复不会被记住。llm 没有 cache_id（测试用的替身）时不存不取。
    ask 区分「同一个请求有意问的第几次」：各次的回复分开存，重跑时每一次用的还是它自己那一份。
    """
    path = None
    cache_id = getattr(llm, "cache_id", None)
    if cache_id:
        key = hashlib.sha256(json.dumps([cache_id, ask, messages], sort_keys=True).encode("utf-8")).hexdigest()
        path = os.path.join(LLM_CACHE_DIR, key[:2], key + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                content = json.load(f)["reply"]
            result = parse(content)
            _dbg("%s reply taken from %s (%d chars): %s" % (label, path, len(content), content[:3000]))
            return result
        except Exception:  # 没存过，或者存下的回复按现在的规则通不过：重新问
            pass
    content = llm.complete(messages)
    _dbg("%s reply (%d chars): %s" % (label, len(content), content[:3000]))
    result = parse(content)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            part = "%s.%d.part" % (path, threading.get_ident())  # 两个线程可能同时存同一个请求的回复
            with open(part, "w", encoding="utf-8") as f:
                json.dump({"reply": content, "saved": datetime.datetime.now().isoformat(timespec="seconds")}, f)
            os.replace(part, path)
        except OSError as e:
            _dbg("[warn] could not save the LLM reply: %r" % str(e)[:200])
    return result


# 窗口小一些，模型对每个词的注意力更集中；上下文约一分钟，隔得较远的重说也看得到
_LLM_WINDOW_WORDS, _LLM_CONTEXT_WORDS, _LLM_WORKERS = 300, 150, 10
_LLM_ATTEMPTS = 3         # 每窗最多判定几次
_LLM_RETRY_WORKERS = 2    # 重试时的并发：失败多半是限流，要放慢
_LLM_RETRY_WAIT_SEC = 5   # 第 n 次重试前等 n 倍这么久
# 一窗删掉超过这个比例就不采用。实测一小时口播（500 词一窗）平均删 14%，最高的窗口 27%；
# 比例很高多半是模型出错（如把所有编号都列出来），误删内容比漏删口误严重得多
_LLM_MAX_DELETE_RATIO = 0.6
_LLM_GUARD_MIN_WORDS = 100  # 词太少的窗口（文件末尾）比例没有意义，不检查
# 复核一窗恢复超过这个比例也不采用：大多数删除应该是对的，几乎全部恢复多半是模型把两个列表的意思弄反了
_LLM_MAX_RESTORE_RATIO = 0.6
_LLM_RESTORE_GUARD_MIN = 10


def _llm_retry_useful(error: Exception) -> bool:
    """限流、服务端错误、超时、回复格式不对都值得重试：一窗失败，这段的口误就留在成片里。"""
    return getattr(error, "status_code", None) not in _HTTP_NO_RETRY


def _timecode(sec: float) -> str:
    s = int(sec)
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def _make_windows(n: int, first: int) -> List[Tuple[int, int, int, int]]:
    """切窗，返回 (上下文起点, 判定起点, 判定终点, 上下文终点)。first 是第一窗判定的词数，其余每窗 _LLM_WINDOW_WORDS 个词。"""
    starts = [0] + list(range(first, n, _LLM_WINDOW_WORDS))
    ends = starts[1:] + [n]
    return [(max(0, a - _LLM_CONTEXT_WORDS), a, b, min(n, b + _LLM_CONTEXT_WORDS)) for a, b in zip(starts, ends)]


def _window_prompt(words: List[Dict], win: Tuple[int, int, int, int], rules: str, closing: str, language: str,
                   proposed: Set[int] = frozenset()) -> str:
    """一个窗口的完整请求内容：规则 + 前文上下文 + 待判定内容 + 后文上下文 + 结尾要求。"""
    ctx_start, a, b, ctx_end = win
    before_note, judge_note, after_note = _WINDOW_NOTES[_norm_language(language)]
    sep = _word_sep(language)
    lines = [judge_note] + _win_lines(words, a, b, sep, proposed)
    if ctx_start < a:
        lines = [before_note] + _win_lines(words, ctx_start, a, sep, proposed) + lines
    if b < ctx_end:
        lines += [after_note] + _win_lines(words, b, ctx_end, sep, proposed)
    return (rules + "\n\n---\n\nTRANSCRIPT (each word on its own line; the LEFT number is that "
            "word's GLOBAL index):\n" + "\n".join(lines) + "\n\n" + closing)


def _check_delete_ratio(deleted: int, judged: int) -> None:
    if judged >= _LLM_GUARD_MIN_WORDS and deleted > _LLM_MAX_DELETE_RATIO * judged:
        raise ValueError("the reply deleted %d of %d words, which looks wrong, so it was not applied" % (deleted, judged))


def _time_range(words: List[Dict], first: int, last: int) -> str:
    """words[first] 到 words[last] 在素材里的时间范围，用来告诉用户哪一段没处理好。"""
    return "%s-%s" % (_timecode(words[first]["start"]), _timecode(words[last]["end"]))


def _run_windows(windows: List[Tuple], judge, where, progress,
                 span: Tuple[float, float], label: str) -> Tuple[Dict, List[Tuple[str, str]]]:
    """并行跑各窗，返回 ({窗口: judge 的结果}, 失败窗口的 (时间范围, 错误))。where(窗口) 给出它的时间范围。

    窗口可以乱序完成：judge 只读同一份输入，只产出自己判定范围内的结果，窗口之间没有依赖，所以完成顺序不影响最终结果。
    失败的窗口等其他窗口跑完后降低并发重试，重试用尽才算失败。
    """
    total = len(windows)
    lo, hi = span

    def _run(win):
        try:
            return win, judge(win), None
        except Exception as e:
            _dbg("[warn] %s window %s failed: %r" % (label, where(win), str(e)[:300]))
            return win, None, e

    results: Dict = {}
    failures: List[Tuple[str, str]] = []
    pending = windows
    done = 0
    for attempt in range(1, _LLM_ATTEMPTS + 1):
        if attempt > 1:
            _report(progress, lo + (hi - lo) * done / total, "%s: retrying %d failed windows…" % (label, len(pending)))
            time.sleep(_LLM_RETRY_WAIT_SEC * (attempt - 1))
        retry = []
        workers = _LLM_WORKERS if attempt == 1 else _LLM_RETRY_WORKERS
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as ex:
            for fut in as_completed([ex.submit(_run, w) for w in pending]):
                win, got, error = fut.result()
                if error is not None and attempt < _LLM_ATTEMPTS and _llm_retry_useful(error):
                    retry.append(win)
                    continue
                if error is None:
                    results[win] = got
                else:
                    failures.append((where(win), str(error)[:200]))
                done += 1
                _report(progress, lo + (hi - lo) * done / total, "%s %d/%d windows…" % (label, done, total))
        pending = retry
        if not pending:
            break
    failures.sort()
    return results, failures


def llm_dedup_words(words: List[Dict], llm: _LLMClient, progress=None, language: str = "en",
                    span: Tuple[float, float] = (0.45, 0.51)) -> Tuple[List[int], List[Tuple[str, str]], int]:
    """第一轮：用 LLM 分窗判定口误，返回 (要删除的全局词索引, 失败窗口的 (时间范围, 错误), 窗口总数)。

    每窗判定 _LLM_WINDOW_WORDS 个词，前后各带 _LLM_CONTEXT_WORDS 个词的只读上下文。
    规则是「保留最后一次、删除更早的」，所以必须看到后文，才能判定窗口末尾的词后面是否被重说；
    只采用待判定段内的索引，上下文两端的半句话就不会被误判成残句。
    """
    n = len(words)
    if n == 0:
        return [], [], 0
    rules, system_prompt = _dedup_prompts(language)
    windows = _make_windows(n, _LLM_WINDOW_WORDS)

    def _judge(win) -> Set[int]:
        _, a, b, _ = win
        user_prompt = _window_prompt(words, win, rules, "Return ONLY a JSON object {\"indices\":[...]} listing the "
                                     "GLOBAL index numbers to DELETE.", language)

        def _parse(content: str) -> Set[int]:
            got = {i for i in _parse_indices(content) if a <= i < b}
            _check_delete_ratio(len(got), b - a)
            return got

        got = _ask(llm, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                   _parse, "window %d-%d" % (a, b))
        _dbg("window %d-%d -> added %d" % (a, b, len(got)))
        return got

    results, failures = _run_windows(windows, _judge, lambda win: _time_range(words, win[1], win[2] - 1),
                                     progress, span, "LLM dedup")
    deleted = set().union(*results.values())
    _dbg("LLM dedup total deleted = %d, failed windows = %d/%d" % (len(deleted), len(failures), len(windows)))
    return sorted(deleted), failures, len(windows)


def llm_review_cuts(words: List[Dict], llm: _LLMClient, proposed: Set[int], locked: Set[int], progress=None,
                    language: str = "en", span: Tuple[float, float] = (0.82, 0.90),
                    label: str = "LLM review") -> Tuple[Set[int], Set[int], List[Tuple[str, str]], int]:
    """复核里由 LLM 判断的部分：按听众听到的内容通读，恢复误删的词，补删漏掉的口误。
    返回 (恢复的词, 补删的词, 失败窗口的 (时间范围, 错误), 窗口总数)。

    words 是听过成片之后的 token 列表（见 _merge_heard）：没标记的是成片里实际听到的词，proposed 是已删除的词。
    所有窗口都读同一份快照，只改自己判定范围内的词，所以和第一轮一样可以乱序执行。
    窗口边界和第一轮错开半窗，第一轮窗口边界附近的词落在复核窗口的中间。
    locked（规则删除的语气词）不能恢复。失败的窗口保持原来的判定。
    """
    rules, system_prompt = _review_prompts(language)
    proposed = frozenset(proposed)
    windows = _make_windows(len(words), _LLM_WINDOW_WORDS // 2)

    def _judge(win) -> Tuple[Set[int], Set[int]]:
        _, a, b, _ = win
        user_prompt = _window_prompt(words, win, rules, "Return ONLY a JSON object {\"restore\":[...],\"delete\":[...]} "
                                     "using GLOBAL index numbers.", language, proposed)
        restorable = {i for i in proposed if a <= i < b and i not in locked}

        def _parse(content: str) -> Tuple[Set[int], Set[int]]:
            restore, delete = _parse_review(content)
            restore &= restorable
            delete = {i for i in delete if a <= i < b and i not in proposed}
            # 只看这次回复自己补删了多少，和还留着的词比：之前各轮已经删掉的不算在它头上，
            # 否则重说很多的段落（已删词本来就超过上限）无论回复什么都会被判成失败
            _check_delete_ratio(len(delete), b - a - sum(1 for i in proposed if a <= i < b))
            if len(restorable) >= _LLM_RESTORE_GUARD_MIN and len(restore) > _LLM_MAX_RESTORE_RATIO * len(restorable):
                raise ValueError("the reply restored %d of %d cuts, which looks wrong, so it was not applied"
                                 % (len(restore), len(restorable)))
            return restore, delete

        return _ask(llm, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    _parse, "review window %d-%d" % (a, b))

    results, failures = _run_windows(windows, _judge, lambda win: _time_range(words, win[1], win[2] - 1),
                                     progress, span, label)
    restored = {i for restore, _ in results.values() for i in restore}
    added = {i for _, delete in results.values() for i in delete}
    _dbg("LLM review: restored %d, deleted %d more, failed windows = %d/%d"
         % (len(restored), len(added), len(failures), len(windows)))
    return restored, added, failures, len(windows)


# ---- 按句判定：画外话和整句重说 ----
# 一行是一句话或一个短语：在句末标点、已有的剪切处、不短于 _LINE_PAUSE_SEC 的停顿处换行。
# 对照人工修正过的工程实测（85 分钟口播，人工在工具保留的词里又删掉 1022 个）：这样分行，整行删除能表达其中的
# 923 个词，只连带 4 个不该删的词；只按标点分行是 775 个和 58 个。再按逗号或大写细分，几乎没有增益
_LINE_PAUSE_SEC = 0.7
_LINE_COMMA_WORDS = 30       # 一行超过这么多词，遇到逗号也换行
_LINE_MAX_WORDS = 60         # 一行最多这么多词（转写没有标点时的保底）
_LINE_PAUSE_NOTE_SEC = 2.0   # 行前的沉默不短于这么久，才在提示词里写出来
# 窗口比逐词判定大：一行只占提示词的一行，而整段重说可能隔着好几句话
_TAKE_WINDOW_WORDS, _TAKE_CONTEXT_WORDS = 600, 300
TAKE_READINGS = 2            # 每一行读几遍（各遍的窗口边界互相错开）。每一遍都认定该剪才剪，否则只做标记
TAKE_MODES = ("cut", "mark", "off")  # 认定的行：剪掉 / 只做标记不剪 / 不做按句判定


def _lines(words: List[Dict], del_set: Set[int]) -> List[List[int]]:
    """把保留的词分成行，返回每行的词下标。"""
    lines: List[List[int]] = []
    cur: List[int] = []
    for i, w in enumerate(words):
        if i in del_set:
            continue
        if cur:
            prev = cur[-1]
            last = words[prev]["word"].rstrip()
            if (last.endswith(_SENTENCE_END) or i != prev + 1
                    or float(w["start"]) - float(words[prev]["end"]) >= _LINE_PAUSE_SEC
                    or len(cur) >= _LINE_MAX_WORDS
                    or (len(cur) >= _LINE_COMMA_WORDS and last.endswith((",", ";", "，", "；", "、")))):
                lines.append(cur)
                cur = []
        cur.append(i)
    if cur:
        lines.append(cur)
    return lines


def _silence_before(words: List[Dict], lines: List[List[int]], n: int) -> float:
    """说第 n 行之前说话人沉默了多久：它和上一行之间的时间，减去中间被删掉的词占的时间。"""
    if n == 0:
        return 0.0
    first, prev = lines[n][0], lines[n - 1][-1]
    gap = float(words[first]["start"]) - float(words[prev]["end"])
    spoken = sum(max(0.0, float(words[i]["end"]) - float(words[i]["start"])) for i in range(prev + 1, first))
    return max(0.0, gap - spoken)


def _line_rows(words: List[Dict], lines: List[List[int]], a: int, b: int, sep: str, language: str) -> List[str]:
    """把第 a 到 b-1 行格式化成提示词里的文本：行号、行前的长停顿、这一行的内容。"""
    note = "（停顿 %d 秒后）" if _norm_language(language) == "zh" else "(after a %d s pause) "
    rows = []
    for n in range(a, b):
        silence = _silence_before(words, lines, n)
        rows.append("L%d  %s%s" % (n, note % round(silence) if silence >= _LINE_PAUSE_NOTE_SEC else "",
                                   sep.join(words[i]["word"].strip() for i in lines[n])))
    return rows


def _line_windows(lines: List[List[int]], first: int) -> List[Tuple[int, int, int, int]]:
    """按词数把行切成窗口，返回 (上下文起点, 判定起点, 判定终点, 上下文终点)，都是行号。
    first 是第一窗判定的词数，其余每窗约 _TAKE_WINDOW_WORDS 个词；上下文前后各约 _TAKE_CONTEXT_WORDS 个词。"""
    bounds, count, target = [0], 0, first
    for n, idxs in enumerate(lines[:-1]):
        count += len(idxs)
        if count >= target:
            bounds.append(n + 1)
            count, target = 0, _TAKE_WINDOW_WORDS
    bounds.append(len(lines))
    windows = []
    for a, b in zip(bounds, bounds[1:]):
        ctx_a, ctx_b, before, after = a, b, 0, 0
        while ctx_a > 0 and before < _TAKE_CONTEXT_WORDS:
            ctx_a -= 1
            before += len(lines[ctx_a])
        while ctx_b < len(lines) and after < _TAKE_CONTEXT_WORDS:
            after += len(lines[ctx_b])
            ctx_b += 1
        windows.append((ctx_a, a, b, ctx_b))
    return windows


def llm_judge_lines(words: List[Dict], del_set: Set[int], llm, progress=None, language: str = "en",
                    span: Tuple[float, float] = (0.51, 0.55)
                    ) -> Tuple[Dict[int, str], Dict[int, str], List[Tuple[str, str]], int]:
    """按句判定：读删掉口误之后剩下的内容，找出整行该剪的：画外话（off）和后面又说了一遍的前一版（retake）。
    返回 ({词下标: 原因} 要剪的, {词下标: 原因} 留下但存疑的, 失败窗口的 (时间范围, 错误), 窗口总数)。

    每一行读 TAKE_READINGS 遍，各遍的窗口边界互相错开，同一行每次都和不同的上下文一起读。
    每一遍都认定该剪的行才剪；只有一部分认定、或者模型自己说拿不准的行留在成片里，标成存疑交给人工确认：
    整句误删很难发现（成片里少一句话并不显眼），漏删一听就知道，所以拿不准时宁可留着。
    前一版要指明取代它的后一行，而且那一行必须留在成片里：同一段内容的各版不会全被剪掉。
    只有一个窗口的短录音，各遍读到的内容完全相同，只是同一个问题问了两次：这样的核对比换一段上下文再读要弱。
    """
    lines = _lines(words, del_set)
    if not lines:
        return {}, {}, [], 0
    rules, system_prompt = _take_prompts(language)
    before_note, judge_note, after_note = _LINE_WINDOW_NOTES[_norm_language(language)]
    sep = _word_sep(language)
    size = [len(idxs) for idxs in lines]
    windows = [(r,) + win for r in range(TAKE_READINGS)
               for win in _line_windows(lines, _TAKE_WINDOW_WORDS * (TAKE_READINGS - r) // TAKE_READINGS)]

    def _judge(win) -> Tuple[Set[int], Dict[int, int], Set[int]]:
        r, ctx_a, a, b, ctx_b = win
        rows = [judge_note] + _line_rows(words, lines, a, b, sep, language)
        if ctx_a < a:
            rows = [before_note] + _line_rows(words, lines, ctx_a, a, sep, language) + rows
        if b < ctx_b:
            rows += [after_note] + _line_rows(words, lines, b, ctx_b, sep, language)
        user_prompt = (rules + "\n\n---\n\nTRANSCRIPT (one numbered line per sentence or phrase):\n" + "\n".join(rows)
                       + "\n\nReturn ONLY a JSON object {\"off_script\":[...],\"earlier_take\":[[...]],\"unsure\":[...]} "
                         "using the line numbers.")

        def _parse(content: str) -> Tuple[Set[int], Dict[int, int], Set[int]]:
            off, pairs, unsure = _parse_takes(content)
            off = {n for n in off if a <= n < b}
            # 取代前一版的那一行必须在它后面，而且是这个窗口里看得到的行
            later = {x: y for x, y in pairs if a <= x < b and x < y < ctx_b and x not in off}
            retake: Dict[int, int] = {}
            for x, y in later.items():
                while y in later:  # 取代它的那一行自己也是前一版：顺着找到最后留下的那一行
                    y = later[y]
                if y not in off:
                    retake[x] = y
            named = {x for x, _ in pairs} | unsure
            unsure = {n for n in named if a <= n < b and n not in off and n not in retake}
            _check_delete_ratio(sum(size[n] for n in off | set(retake)), sum(size[a:b]))
            return off, retake, unsure

        # 短录音只有一个窗口时，各遍的请求一字不差：按第几遍分开存，重跑时两遍用的还是各自的回复
        return _ask(llm, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    _parse, "line window %d/L%d-L%d" % (r + 1, a, b), ask=r)

    results, failures = _run_windows(
        windows, _judge, lambda win: _time_range(words, lines[win[2]][0], lines[win[3] - 1][-1]),
        progress, span, "Line check")
    # 每一遍认定该剪的行：行号 -> 取代它的那一行（画外话没有，记 None）
    sure: List[Dict[int, Optional[int]]] = [{} for _ in range(TAKE_READINGS)]
    maybe: Set[int] = set()
    for win, (off, retake, unsure) in results.items():
        sure[win[0]].update(dict.fromkeys(off))
        sure[win[0]].update(retake)
        maybe |= unsure
    agreed = set(sure[0]).intersection(*sure[1:])

    def _replaced(n: int) -> bool:
        """第 n 行是前一版时，至少有一遍给出的「取代它的那一行」留在成片里。"""
        for reading in sure:
            y = reading[n]
            while y is not None and y in agreed and reading.get(y) is not None:
                y = reading[y]
            if y is not None and y not in agreed:
                return True
        return False

    cut_lines = {n for n in agreed if any(reading[n] is None for reading in sure) or _replaced(n)}
    cut: Dict[int, str] = {}
    flag: Dict[int, str] = {}
    for n in sorted(set().union(maybe, *sure)):
        whys = ["off" if reading[n] is None else "retake" for reading in sure if n in reading]
        why = "unsure" if not whys else "off" if "off" in whys else "retake"
        target = cut if n in cut_lines else flag
        target.update(dict.fromkeys(lines[n], why))
        _dbg("line check %s [%s] L%d @%s: %s" % ("cut" if n in cut_lines else "flag", why, n,
                                                 _timecode(words[lines[n][0]]["start"]),
                                                 sep.join(words[i]["word"].strip() for i in lines[n])[:200]))
    _dbg("line check: %d lines, cut %d (%d words), flagged %d (%d words), failed windows = %d/%d"
         % (len(lines), len(cut_lines), len(cut), len(set().union(maybe, *sure)) - len(cut_lines), len(flag),
            len(failures), len(windows)))
    return cut, flag, failures, len(windows)


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


def _quiet_level(env: np.ndarray, lo: int, hi: int) -> Optional[float]:
    """[lo, hi) 附近低于多少算安静。门限按前后 5 秒的音量定，适应不同音量和底噪；
    响度几乎不变（持续噪声等）、分不出哪里是空隙时返回 None。"""
    around = env[max(0, lo - 1000):hi + 1000]
    floor, speech = np.percentile(around, 5), np.percentile(around, 95)
    if speech - floor < _MIN_CONTRAST_DB:
        return None
    return float(max(floor + 8, speech - 28))


def _valleys(env: np.ndarray, lo: int, hi: int) -> List[Tuple[int, int]]:
    """[lo, hi) 里所有够长的安静段。"""
    level = _quiet_level(env, lo, hi)
    if level is None:
        return []
    quiet = env[lo:hi] <= level
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


# 听成片时听出来的残音（firm）和旁边的保留词之间，这么近以内的空隙才算两者的分界
_FIRM_REACH_SEC = 0.15


def _start_after(env: np.ndarray, leak_end: float, start: float) -> float:
    """保留词前面紧挨着听出来的残音时，块从哪里开始：残音后面紧跟着空隙就从空隙的末尾开始，连在一起就从残音结束处开始。

    不能像 _sound_start 那样找离保留词时间戳最近的空隙：复核时听到的时间戳偶尔会晚几百毫秒
    （实测 "frequency" 晚了 0.3 秒），离它最近的空隙就落在词的中间，切点会切掉词的前半截。
    """
    lo = _env_frame(env, leak_end)
    hi = _env_frame(env, max(start, leak_end) + _VALLEY_REACH_SEC)
    near = _env_frame(env, leak_end + _FIRM_REACH_SEC)
    valleys = [v for v in _valleys(env, lo, min(len(env), hi)) if v[0] <= near] if hi > lo else []
    if not valleys:
        return leak_end
    return float(max(leak_end, min(valleys[0][1] / _ENV_RATE - _LEAD_SEC, start)))


def _end_before(env: np.ndarray, leak_start: float, end: float) -> float:
    """保留词后面紧挨着听出来的残音时，块到哪里结束：和 _start_after 对称。"""
    hi = _env_frame(env, leak_start)
    lo = _env_frame(env, min(end, leak_start) - _VALLEY_REACH_SEC)
    near = _env_frame(env, leak_start - _FIRM_REACH_SEC)
    valleys = [v for v in _valleys(env, lo, hi) if v[1] >= near] if hi > lo else []
    if not valleys:
        return leak_start
    return float(min(leak_start, max(valleys[-1][0] / _ENV_RATE + _TAIL_SEC, end)))


# 被切断的词：从原切点往外重新找空隙。原切点常常就落在词内的塞音上（"plat|form"、"cap|ital"），
# 所以要先跳过一小段；找不到空隙（和下一个词连读）时固定外扩，下一轮听成片时再确认
_CLIP_SKIP_SEC = 0.06
_CLIP_REACH_SEC = 0.6
_CLIP_FALLBACK_SEC = 0.15


def _gap_after(env: np.ndarray, t: float) -> float:
    """t 之后的第一个空隙（词在 t 被切断时，它真正结束的位置）。"""
    lo = _env_frame(env, t + _CLIP_SKIP_SEC)
    hi = _env_frame(env, t + _CLIP_REACH_SEC)
    valleys = _valleys(env, lo, min(len(env), hi + int(_VALLEY_REACH_SEC * _ENV_RATE))) if hi > lo else []
    if not valleys or valleys[0][0] >= hi:
        return t + _CLIP_FALLBACK_SEC
    a, b = valleys[0]
    return float(min(a + _TAIL_SEC * _ENV_RATE, b) / _ENV_RATE)


def _gap_before(env: np.ndarray, t: float) -> float:
    """t 之前的最后一个空隙（词在 t 被切掉开头时，它真正开始的位置）。"""
    lo = _env_frame(env, t - _CLIP_REACH_SEC)
    hi = _env_frame(env, t - _CLIP_SKIP_SEC)
    valleys = _valleys(env, max(0, lo - int(_VALLEY_REACH_SEC * _ENV_RATE)), hi) if hi > lo else []
    if not valleys or valleys[-1][1] <= lo:
        return max(0.0, t - _CLIP_FALLBACK_SEC)
    a, b = valleys[-1]
    return float(max(b - _LEAD_SEC * _ENV_RATE, a) / _ENV_RATE)


def _cut_in_sound(env: np.ndarray, t: float) -> bool:
    """切点 t 是否落在声音里（它不在任何空隙里：两边的词连读，或者切点没能校准到空隙上）。
    响度分不出空隙时返回 False：没有依据，就不当成落在声音里。"""
    at = _env_frame(env, t)
    reach = int(_VALLEY_REACH_SEC * _ENV_RATE)
    lo, hi = max(0, at - reach), min(len(env), at + reach)
    if hi <= lo or _quiet_level(env, lo, hi) is None:
        return False
    # 切点取整到毫秒再换算成帧，可能比空隙的边缘偏出一帧
    return not any(a - 1 <= at <= b + 1 for a, b in _valleys(env, lo, hi))


def _same_run(prev: Dict, cur: Dict, gap: float, keep_gap: float) -> bool:
    """两个相邻的保留词是否留在同一块里。

    听过成片之后，词带着它在那一版成片里的块号（block）：同一块的不拆，不同块的不并，
    没改动的地方才会和听过的那一版完全一样。没有块号的词（刚恢复的词等）按停顿长短判断。
    """
    prev_block, cur_block = prev.get("block"), cur.get("block")
    if prev_block is not None and cur_block is not None:
        return prev_block == cur_block
    return gap <= keep_gap


def _keep_block_details(words: List[Dict], del_set: Set[int], keep_gap: float = 0.4, pad: float = 0.06,
                        env: Optional[np.ndarray] = None) -> List[Dict]:
    """从「保留词」构建保留块，返回每块的 first/last（词下标）和 start/end（秒），合并之前的样子。

    被删的词会在中途切断块（不跨越删除区合并），保留块内部保留 <=keep_gap 的自然停顿。
    块的两端先按时间戳加 pad，且 pad 不越过相邻被删词的边界。传入 env（响度包络）时，再把两端
    校准到真实的声音空隙上，保住保留词的尾音，也不带进被删词的尾音；找不到空隙的一端保持不变。

    听过成片之后，词上可以带这些标记（见 _merge_heard）：
      block       这个词在听过的那一版成片里属于第几块，见 _same_run
      lead / tail 这个词是块的第一个 / 最后一个词时，块从哪里开始 / 到哪里结束：听过没问题的切点原样保留，
                  被切断的词换成重新找的位置
      firm        这个被删词的位置是听出来的，旁边的切点由它决定（_start_after / _end_before），不再看保留词的时间戳
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
        elif _same_run(words[last], w, s - cur_e, keep_gap):
            last, cur_e = i, max(cur_e, e)
        else:
            runs.append((first, last, cur_e))
            first, last, cur_e = i, i, e
    if first is not None:
        runs.append((first, last, cur_e))
    del_spans.sort()

    def _firm(i: int) -> bool:
        return 0 <= i < len(words) and i in del_set and bool(words[i].get("firm"))

    blocks: List[Dict] = []
    for a, b, e in runs:
        s = float(words[a]["start"])
        lead, tail = words[a].get("lead"), words[b].get("tail")
        if e - s <= 0.02 and lead is None and tail is None:
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
        start = end = None
        if lead is not None:
            ps = start = float(lead)
        elif env is not None and _firm(a - 1):
            ps = start = _start_after(env, float(words[a - 1]["end"]), s)
        elif env is not None:
            start = _sound_start(env, s, float(words[a - 1]["end"]) if a > 0 else 0.0)
        if tail is not None:
            pe = end = float(tail)
        elif env is not None and _firm(b + 1):
            pe = end = _end_before(env, float(words[b + 1]["start"]), e)
        elif env is not None:
            end = _sound_end(env, e, float(words[b + 1]["start"]) if b + 1 < len(words) else float("inf"))
        blocks.append({"first": a, "last": b, "stamp_start": ps, "stamp_end": pe,
                       "start": ps if start is None else start, "end": pe if end is None else end})
    # 很短的被删词两侧的切点可能校准到一起，两块一合并，被删词就留下来了：这一处退回时间戳
    for prev, cur in zip(blocks, blocks[1:]):
        if cur["first"] > prev["last"] + 1 and cur["start"] <= prev["end"]:
            prev["end"], cur["start"] = prev["stamp_end"], cur["stamp_start"]
    return [block for block in blocks if block["end"] > block["start"]]


def _merge_blocks(blocks: List[Dict]) -> List[Tuple[float, float]]:
    """合并相接/重叠的相邻块，避免拼接时同一段被重复（叠音/没对齐）。"""
    merged: List[Tuple[float, float]] = []
    for block in blocks:
        s, e = block["start"], block["end"]
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def clean_keep_blocks(words: List[Dict], del_set: Set[int], keep_gap: float = 0.4, pad: float = 0.06,
                      env: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
    """直接从「保留词」构建保留块（见 _keep_block_details），合并相接/重叠的块后返回 (起点, 终点) 列表。"""
    return _merge_blocks(_keep_block_details(words, del_set, keep_gap, pad, env))


# ---- 第二轮：听成片 ----
# 第一轮只看过转写文字。实测一小时口播的成片里，约五分之一的接缝听起来和计划的不一样：
#   * 被删词的时间戳偏了，它的头或尾漏进了成片（"less then the less upkeep"）；
#   * 保留词的时间戳偏早，或者切点落在词内的塞音上，词被切断（"situation" 只剩 "situa-"）；
#   * 第一次转写漏掉了一段重说（"You're gonna ne- you're gonna need"），第一轮根本没见过它。
# 这些只有听成片才能发现：把成片拼出来重新逐字转写，和计划保留的词对齐，再把差异变成对剪辑的修改。
_ALIGN_SLACK_SEC = 0.1     # 对齐时，听到的词可以比它那一块的范围超出这么多（转写时间戳在接缝处有误差）
_LEAK_MAX_SEC = 1.0        # 接缝处漏进来的声音最长算这么多；更长的多半是第一次转写漏掉的一段话，交给 LLM 判断
_CLIP_MIN_CHARS = 2        # 听到的残片至少这么长，才用来判断「词被切断」
_UNHEARD_MIN_CHARS = 4     # 块边缘的词完全没听到时，这么长的词才算被切断：短的虚词常被转写漏掉，不代表没有声音
# 中文按字切分，一个字是一个音节，相当于英文的两三个字母，所以两个门槛按字数减半。
# 多数 token 只有一个字，按长度还是判不出来：这种字要切点落在声音里（_cut_in_sound）才算被切断
_CLIP_MIN_CHARS_ZH = 1
_UNHEARD_MIN_CHARS_ZH = 2
REVIEW_PASSES = 2          # 默认听几遍：第一遍发现并修改，第二遍确认修改后的成片


def _edit_tokens(words: List[Dict], proposed: Set[int], locked: Set[int], takes: Optional[Dict[int, str]] = None,
                 flags: Optional[Dict[int, str]] = None) -> List[Dict]:
    """把转写的词变成带剪辑标记的 token：cut 记录它是被谁删掉的（None 表示保留），locked 表示复核不能恢复。

    takes 是按句判定整行剪掉的词（cut 记成原因，见 _TAKE_CUTS）：它们读过 TAKE_READINGS 遍才定下来，复核不再恢复。
    flags 是留在成片里但存疑的词（flag 记成原因）：不影响剪辑，只在剪辑清单和 Premiere 工程里标出来。
    """
    takes, flags = takes or {}, flags or {}
    tokens = []
    for i, w in enumerate(words):
        cut = "first" if i in proposed else takes.get(i)
        token = {"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                 "cut": cut, "locked": i in locked or cut in _TAKE_CUTS}
        if not cut and i in flags:
            token["flag"] = flags[i]
        tokens.append(token)
    return tokens


def _cut_indices(tokens: List[Dict]) -> Set[int]:
    return {i for i, t in enumerate(tokens) if t.get("cut")}


def _plan_edit(tokens: List[Dict], keep_gap: float, env: Optional[np.ndarray],
               total_dur: float) -> Tuple[List[Tuple[float, float]], _EditTimeline, Dict[int, int]]:
    """按当前的删除标记排出成片，返回 (保留块, 成片时间轴, {保留词下标: 它在成片的第几块})。"""
    details = _keep_block_details(tokens, _cut_indices(tokens), keep_gap=keep_gap, pad=0.06, env=env)
    keep = _merge_blocks(details)
    timeline = _EditTimeline.of_splice(keep, total_dur)
    word_block: Dict[int, int] = {}
    for block in details:
        mid = (block["start"] + block["end"]) / 2
        b = timeline.block_of_source(mid)
        if b < 0 or mid > timeline.blocks[b][1]:
            continue  # 这一块太短，拼接时被丢掉了，成片里没有它
        for i in range(block["first"], block["last"] + 1):
            word_block[i] = b
    return keep, timeline, word_block


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """两个词在时间上是否基本重合：重叠的部分超过较短那个词的一半。"""
    return min(a1, b1) - max(a0, b0) >= 0.5 * max(0.02, min(a1 - a0, b1 - b0))


def _is_cut_off(word: str) -> bool:
    """转写把说到一半就断掉的词写成以连字符结尾（"situa-"、"我—"）。"""
    return bool(re.search(r"\w[-—]\W*$", word))


def _merge_heard(tokens: List[Dict], word_block: Dict[int, int], heard: List[Dict], timeline: _EditTimeline,
                 env: np.ndarray, language: str, last_pass: bool = False) -> Tuple[List[Dict], collections.Counter]:
    """把成片里听到的词（heard，成片时间）并进 token 列表，返回 (新的 token 列表, 各类发现的数量)。

    计划保留的词和听到的词按文字对齐，然后分情况处理：
      * 对上了：词照旧，时间换成这次听到的位置（换算回素材时间）。
      * 听成了别的词：保留原来的词。两次转写本来就有约 0.5% 的词不一样，不能据此改剪辑。
        但块边缘的词只听到一半（或完全没听到）说明它被切断了，这一侧的切点往外挪到下一个空隙。
      * 只在成片里听到、计划里没有的词：加进列表，默认保留，交给 LLM 判断。其中语气词直接删；
        紧贴一处剪切、排在这一块所有计划词之外的，是被删词漏进来的残音，也删掉（LLM 可以恢复）。
      * 上一遍在某个词旁边修掉了声音，这一遍这个词听不到了：修掉的其实是它的一部分，撤销那次修剪。
    没有问题的切点记在词上（lead / tail），重新排成片时原样使用，所以没改动的地方和听过的这一版完全相同。
    last_pass 表示这一遍之后不会再听：把握不大的修改留给还有下一遍的时候做。
    """
    found: collections.Counter = collections.Counter()
    keys = [_as_posix_clean(t["word"]) for t in tokens]
    hkeys = [_as_posix_clean(h["word"]) for h in heard]
    planned = sorted(word_block)
    n_blocks = len(timeline.blocks)
    block_tokens: Dict[int, List[int]] = collections.defaultdict(list)
    for i in planned:
        block_tokens[word_block[i]].append(i)

    # 1) 按文字对齐。只接受落在同一块里的配对：文字相同但不在一块的，是别处碰巧说了同样的词
    match: Dict[int, int] = {}  # 听到的词 -> 计划保留的词
    matcher = difflib.SequenceMatcher(None, [keys[i] for i in planned], hkeys, autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            i, j = planned[a + k], b + k
            lo, hi = timeline.offsets[word_block[i]], timeline.offsets[word_block[i] + 1]
            if hkeys[j] and heard[j]["end"] > lo - _ALIGN_SLACK_SEC and heard[j]["start"] < hi + _ALIGN_SLACK_SEC:
                match[j] = i
    heard_of = {i: j for j, i in match.items()}

    # 2) 每个听到的词属于成片的哪一块：对上的跟着计划的词走，没对上的按时间定，但不越过前后已对上的词
    next_anchor = [len(tokens)] * (len(heard) + 1)  # 它之后最近一个对上的词（token 下标）
    for j in range(len(heard) - 1, -1, -1):
        next_anchor[j] = match.get(j, next_anchor[j + 1])
    hblock: List[int] = []
    prev_anchor: List[int] = []                     # 它之前最近一个对上的词
    floor, anchor = 0, -1
    for j, h in enumerate(heard):
        prev_anchor.append(anchor)
        if j in match:
            anchor = match[j]
            floor = word_block[anchor]
        else:
            ceil = word_block[next_anchor[j]] if next_anchor[j] < len(tokens) else n_blocks - 1
            floor = max(floor, min(ceil, timeline.block_of_edit((h["start"] + h["end"]) / 2)))
        hblock.append(floor)
    span = [(timeline.to_source(h["start"], b), timeline.to_source(h["end"], b)) for h, b in zip(heard, hblock)]

    # 3) 没对上的词：前后两个对上的词之间，计划的词也有没对上的，就是同一段声音的另一种听法（variants 记下
    #    时间重合的那一对，用来判断词是否被切断）；计划的词都对上了还多出来的，才是只在成片里听到的词
    variants: Dict[int, List[int]] = collections.defaultdict(list)  # 计划的词 -> 它被听成的词
    extra_by_block: Dict[int, List[int]] = collections.defaultdict(list)
    place: Dict[int, Tuple[int, int]] = {}  # 只在成片里听到的词放在哪：(token 下标, 0 在它之前 / 2 在它之后)
    heard_in: Dict[int, List[int]] = collections.defaultdict(list)  # 每一块里听到的词
    for j in range(len(heard)):
        b = hblock[j]
        if not hkeys[j] or b not in block_tokens:
            continue
        heard_in[b].append(j)
        if j in match:
            continue
        x, y = span[j]
        between = [i for i in block_tokens[b] if prev_anchor[j] < i < next_anchor[j] and i not in heard_of]
        for i in between:
            if _overlap(x, y, tokens[i]["start"], tokens[i]["end"]):
                variants[i].append(j)
        if between:
            # 这一段里计划的词也没对上，两边是同一段声音的两种听法（"gonna" / "going to"、"you have" /
            # "you're happy with"），多出来的那个词不是多出来的声音：把它当真，LLM 删它时切掉的就是计划保留的词
            continue
        # 放在它后面那个对上的词之前；那个词已经在下一块了，就放在这一块的最后
        after = next_anchor[j]
        place[j] = (after, 0) if word_block.get(after) == b else (block_tokens[b][-1], 2)
        extra_by_block[b].append(j)

    # 4) 只在成片里听到的词：语气词直接删；紧贴剪切处、排在这一块所有计划词之外的，是被删词漏进来的残音
    first_of = {b: idxs[0] for b, idxs in block_tokens.items()}
    last_of = {b: idxs[-1] for b, idxs in block_tokens.items()}

    def _cuts_between(lo: int, hi: int) -> List[int]:
        return [i for i in range(lo, hi) if tokens[i].get("cut")]

    fillers = _core_fillers_for(language)
    new_cut: Dict[int, str] = {}     # 只在成片里听到、要删掉的词 -> 原因
    superseded: Dict[int, int] = {}  # 原来的被删词 -> 取代它的那个听到的词
    for b, js in extra_by_block.items():
        cuts_before = _cuts_between(last_of[b - 1] + 1 if b - 1 in last_of else 0, first_of[b])
        cuts_after = _cuts_between(last_of[b] + 1, first_of[b + 1] if b + 1 in first_of else len(tokens))
        leading = [j for j in js if place[j] == (first_of[b], 0)]
        trailing = [j for j in js if place[j] == (last_of[b], 2)]
        # 每个接缝只修一次。修过的地方下一遍还听到多余的声音，多半是上次的切口让转写听错了（实测 "station | and"
        # 修过之后被听成 "dot. and"）：再修就是一遍遍往保留词里切，所以这些词不算数
        for run, flag, i in ((leading, "trimmed_lead", first_of[b]), (trailing, "trimmed_tail", last_of[b])):
            if run and tokens[i].get(flag):
                for j in run:
                    del place[j]
                js[:] = [j for j in js if j not in run]
                del run[:]
        # 从紧挨剪切处的那个词开始，两边成对比较
        for run, pairs in ((leading, zip(reversed(leading), reversed(cuts_before))), (trailing, zip(trailing, cuts_after))):
            if not run or not (cuts_before if run is leading else cuts_after):
                continue
            if span[run[-1]][1] - span[run[0]][0] > _LEAK_MAX_SEC:
                continue
            for j in run:
                new_cut[j] = "leak"
            # 漏进来的词和紧挨着的被删词是同一个词时，以听到的为准：这才是那个词真正的位置
            for j, i in pairs:
                if hkeys[j] != keys[i]:
                    break
                superseded[i] = j
        for j in js:
            if _is_vocal_sound(heard[j]["word"], heard[j - 1]["word"] if j else "", fillers):
                new_cut[j] = "filler"
    found["leaks"] = sum(1 for reason in new_cut.values() if reason == "leak")
    found["fillers"] = sum(1 for reason in new_cut.values() if reason == "filler")
    found["extra"] = len(place) - len(new_cut)
    # 修掉声音的地方在两个保留词上做记号：trimmed_* 表示这个接缝修过了，以后不再修；
    # check_* 表示这个词紧挨着被修掉的声音，下一遍要确认它还听得到
    marks: Dict[int, Set[str]] = collections.defaultdict(set)
    for b, js in extra_by_block.items():
        for j in js:
            if j not in new_cut:
                continue
            i, side = place[j]
            at = block_tokens[b].index(i)
            if side == 0:
                before, after = (block_tokens[b][at - 1] if at else last_of.get(b - 1)), i
                checked = {after} | ({before} if at else set())  # 块中间切掉一段时，前后两个词都紧挨着它
            else:
                before, after, checked = i, first_of.get(b + 1), {i}
            for word, is_lead in ((before, False), (after, True)):
                if word is not None:
                    marks[word].add("trimmed_lead" if is_lead else "trimmed_tail")
                    if word in checked:
                        marks[word].add("check_lead" if is_lead else "check_tail")

    # 5) 上一遍在这个词旁边修掉了声音，这一遍它却听不到了：修掉的不是残音，是转写把这个词拆成了两半
    #    （"station" 被听成 "station dot"）。把那一截还给它，切点回到修之前的位置
    given_back: Dict[Tuple[int, int], List[int]] = {}  # (保留词, 0 词头 / 2 词尾) -> 还给它的 token，由近到远
    for b in block_tokens:
        for i, side, step in ((first_of[b], 0, -1), (last_of[b], 2, 1)):
            if i in heard_of or not tokens[i].get("check_tail" if side else "check_lead"):
                continue
            k, run = i + step, []
            while 0 <= k < len(tokens) and tokens[k].get("cut") and tokens[k].get("heard") and tokens[k].get("firm"):
                run.append(k)
                k += step
            if run:
                given_back[(i, side)] = run
    returned = {k for run in given_back.values() for k in run}
    found["undone"] = len(given_back)

    # 块边缘的词被切断：只听到它的一头，或者整个词都没听到。同一侧只挪一次，挪过还听不到就不再动
    zh = _norm_language(language) == "zh"
    clip_min, unheard_min = (_CLIP_MIN_CHARS_ZH, _UNHEARD_MIN_CHARS_ZH) if zh else (_CLIP_MIN_CHARS, _UNHEARD_MIN_CHARS)

    def _clipped(i: int, at_end: bool) -> bool:
        if (i, 2 if at_end else 0) in given_back:
            return False
        if i in heard_of or tokens[i].get("widened_tail" if at_end else "widened_lead"):
            return False
        if i not in variants:
            # 没听到不一定是被切断（转写偶尔会漏词），挪错了要靠下一遍听出来，所以最后一遍不做
            if last_pass:
                return False
            if len(keys[i]) >= unheard_min:
                return True
            # 中文的单个字按长度判不出来：切点在空隙里，声音是完整的，没听到只是转写漏了；
            # 切点落在声音里、这个字又完全没听到，才是被切断了
            s, e = timeline.blocks[word_block[i]]
            return zh and _cut_in_sound(env, e if at_end else s)
        j = variants[i][-1 if at_end else 0]
        if j != heard_in[word_block[i]][-1 if at_end else 0]:
            return False  # 残片外侧还听到了别的词，词不是在切点上断的
        part, whole = hkeys[j], keys[i]
        if not clip_min <= len(part) < len(whole):
            return False
        if not at_end:
            return whole.endswith(part)
        # 转写把说到一半的词写成 "gra-"，拼写不一定是原词的前缀（great）
        cut_off = _is_cut_off(heard[j]["word"]) and not _is_cut_off(tokens[i]["word"]) and part[0] == whole[0]
        return cut_off or whole.startswith(part)

    lead_fix: Dict[int, float] = {}
    tail_fix: Dict[int, float] = {}
    for b in block_tokens:
        s, e = timeline.blocks[b]
        extras = [place[j] for j in extra_by_block.get(b, [])]
        if (first_of[b], 0) not in extras and _clipped(first_of[b], at_end=False):
            lead_fix[b] = max(_gap_before(env, s), timeline.blocks[b - 1][1] if b > 0 else 0.0)
        if (last_of[b], 2) not in extras and _clipped(last_of[b], at_end=True):
            tail_fix[b] = min(_gap_after(env, e), timeline.blocks[b + 1][0] if b + 1 < n_blocks else float("inf"))
    found["widened"] = len(lead_fix) + len(tail_fix)

    # 6) 排出新的列表。上一遍只在成片里听到、这一遍又听不到的词不再保留：那多半是转写自己编出来的
    items: List[Tuple[Tuple[int, int, int], Dict]] = []
    for i, t in enumerate(tokens):
        unheard = i in word_block and i not in heard_of and i not in variants
        if i in superseded or i in returned or (unheard and t.get("heard")):
            continue
        # 保留词的切点下面重新记；被删的词留着它当初的切点，以后恢复它时就回到听过的那个位置
        stale = ("block", "check_lead", "check_tail") + (() if t.get("cut") else ("lead", "tail"))
        new = {k: v for k, v in t.items() if k not in stale}
        new.update((flag, True) for flag in marks.get(i, ()))
        if i in word_block:
            new["block"] = word_block[i]
            found["unheard"] += unheard
            if i in heard_of:
                new["start"], new["end"] = span[heard_of[i]]
            for side in (0, 2):
                for k in given_back.get((i, side), []):
                    new["start"], new["end"] = min(new["start"], tokens[k]["start"]), max(new["end"], tokens[k]["end"])
        items.append(((i, 1, 0), new))
    replaces = {j: i for i, j in superseded.items()}
    for j, (i, side) in place.items():
        reason = new_cut.get(j)
        new = {"word": heard[j]["word"], "start": span[j][0], "end": span[j][1], "block": hblock[j], "heard": True,
               "cut": reason, "locked": reason == "filler", "firm": bool(reason)}
        if j in replaces:  # 取代了原来的被删词：它是谁删的、能不能恢复，都照原来的
            old = tokens[replaces[j]]
            new["cut"], new["locked"] = old["cut"], new["locked"] or bool(old.get("locked"))
        items.append(((i, side, j), new))
    items.sort(key=lambda item: item[0])
    merged = [t for _, t in items]

    # 7) 记下每一块的两端：没问题的切点原样保留，被切断的一侧换成重新找的位置
    members: Dict[int, List[Dict]] = collections.defaultdict(list)
    for t in merged:
        if t.get("block") is not None:
            members[t["block"]].append(t)
    for b, ts in members.items():
        ts[0]["lead"] = lead_fix.get(b, timeline.blocks[b][0])
        ts[-1]["tail"] = tail_fix.get(b, timeline.blocks[b][1])
        if b in lead_fix:
            ts[0]["widened_lead"] = True
        if b in tail_fix:
            ts[-1]["widened_tail"] = True
        # 还回去的那一截：切点用最外面那个 token 当初记下的位置；没记过就重新按声音找
        for t, key, run in ((ts[0], "lead", given_back.get((first_of[b], 0))),
                            (ts[-1], "tail", given_back.get((last_of[b], 2)))):
            if run:
                t.pop(key)
                if tokens[run[-1]].get(key) is not None:
                    t[key] = tokens[run[-1]][key]
    return merged, found


def review_edit(audio_path: str, tokens: List[Dict], llm: _LLMClient, *, whisper_model: str, language: str,
                keep_gap: float, env: np.ndarray, total_dur: float, passes: int, work_dir: str, progress=None,
                span: Tuple[float, float] = (0.55, 0.90)) -> Tuple[List[Dict], collections.Counter, List[str]]:
    """第二轮：听成片，返回 (复核后的 token 列表, 各类修改的数量, 警告)。

    每一遍：按当前的删除标记拼出成片 → CrisperWhisper 重新逐字转写 → 和计划保留的词对齐（_merge_heard），
    残音、语气词、被切断的词按规则处理 → LLM 读成片里实际听到的内容，恢复误删的词、补删漏掉的口误。
    这一遍改过的地方要到下一遍才听得到，所以默认听两遍；某一遍什么都没改，成片就已经确认，提前结束。
    第一遍听不了（转写失败）时退回只读文字的复核，后面的遍听不了就停在上一遍的结果，都会给出警告。
    """
    warnings: List[str] = []
    totals: collections.Counter = collections.Counter()
    lo, hi = span
    for n in range(1, passes + 1):
        p0, p1 = lo + (hi - lo) * (n - 1) / passes, lo + (hi - lo) * n / passes
        p_render, p_heard = p0 + 0.05 * (p1 - p0), p0 + 0.80 * (p1 - p0)
        label = "ASR review %d/%d" % (n, passes) if passes > 1 else "ASR review"
        keep, timeline, word_block = _plan_edit(tokens, keep_gap, env, total_dur)
        if not keep:
            break
        listened = False
        found: collections.Counter = collections.Counter()
        draft = os.path.join(work_dir, "review%d.flac" % n)
        try:
            _report(progress, p0, label + ": rendering the edit…")
            _splice_keep(audio_path, keep, draft, out_format="flac", work_dir=work_dir, progress=progress,
                         span=(p0, p_render), label=label + ": rendering the edit")
            heard = transcribe(draft, whisper_model, language=language, progress=progress, work_dir=work_dir,
                               span=(p_render, p_heard), stage=label + ": ", out_name="heard%d.json" % n)
            tokens, found = _merge_heard(tokens, word_block, heard, timeline, env, language, last_pass=n == passes)
            listened = True
            _dbg("%s: heard %d words in %d blocks -> %s" % (label, len(heard), len(keep), dict(found)))
        except Exception as e:
            _dbg("[warn] %s could not listen to the edit: %r" % (label, str(e)[:300]))
            if n == 1:
                warnings.append("The edit could not be listened to, so the review only read the transcript and "
                                "leftover sounds or clipped words at the cuts were not checked: %s" % str(e)[:200])
            else:
                warnings.append("Review pass %d could not listen to the edit, so the changes made by pass %d were "
                                "not checked: %s" % (n, n - 1, str(e)[:200]))
        finally:
            try:
                os.remove(draft)
            except OSError:
                pass
        if not listened and n > 1:
            break
        _report(progress, p_heard, label + ": checking what was heard…")
        restored, added, failures, windows = llm_review_cuts(
            tokens, llm, _cut_indices(tokens), {i for i, t in enumerate(tokens) if t.get("locked")},
            progress=progress, language=language, span=(p_heard, p1), label=label + ": LLM")
        for i in restored:
            tokens[i].update(cut=None, restored=True)
        for i in added:
            tokens[i]["cut"] = "review"
        if failures:
            warnings.append("The LLM review failed for %d of %d windows in pass %d, so the cuts in these parts of the "
                            "source were not double-checked: %s. First error: %s"
                            % (len(failures), windows, n, ", ".join(where for where, _ in failures), failures[0][1]))
        found.update(restored=len(restored), deleted=len(added), listened=listened)
        totals.update(found)
        # 这一遍改过的地方还没有听过；下一遍会听到，最后一遍的留在统计里如实说明
        totals["unchecked"] = (found["leaks"] + found["fillers"] + found["widened"] + found["undone"]
                               + len(restored) + len(added))
        if not listened or not totals["unchecked"]:
            break
    return tokens, totals, warnings


def _review_summary(totals: collections.Counter) -> str:
    """一句话说明听成片这一轮做了什么，给页面和剪辑清单用。"""
    if not totals["listened"]:
        return ""

    def _count(n: int, noun: str) -> str:
        return "%d %s%s" % (n, noun, "" if n == 1 else "s")

    text = ("ASR review listened to the edit %s: trimmed %s of deleted words and %s, widened %s that clipped a word%s; "
            "the LLM restored %s and deleted %d more."
            % (_count(totals["listened"], "time"), _count(totals["leaks"], "leftover"),
               _count(totals["fillers"], "vocal sound"), _count(totals["widened"], "cut"),
               ", undid %s that had cut into a kept word" % _count(totals["undone"], "trim") if totals["undone"] else "",
               _count(totals["restored"], "word"), totals["deleted"]))
    if totals["unchecked"]:
        text += " The last pass still made %s, which were not listened to again." % _count(totals["unchecked"], "change")
    return text


def _check_points(tokens: List[Dict], language: str = "en") -> List[Dict]:
    """值得人工先看的地方，按时间排序：整行剪掉的（kind 为 "cut"）和留在成片里但存疑的（kind 为 "flag"）。

    每一项有 kind、why（给人看的原因）、start / end（素材秒）和 text。相邻的同类行合成一项；
    中间夹着的其他被删词（口误、残音）不算间隔，也不进 text。
    存疑片段的 start / end 取它和前后词之间的中点：Premiere 工程在这里把片段切开，
    用户删掉这一段之后，它前后的停顿各留下一半。
    """
    sep = _word_sep(language)
    groups: List[Tuple[str, List[int]]] = []
    current: Optional[Tuple[str, List[int]]] = None
    for i, t in enumerate(tokens):
        cut = t.get("cut")
        kind = "cut" if cut in _TAKE_CUTS else "flag" if not cut and t.get("flag") else None
        if kind is None:
            if not cut:  # 正常保留的词把前后隔开
                current = None
            continue
        if current is None or current[0] != kind:
            current = (kind, [])
            groups.append(current)
        current[1].append(i)

    points = []
    for kind, idxs in groups:
        a, b = idxs[0], idxs[-1]
        whys = {tokens[i]["cut"] if kind == "cut" else tokens[i]["flag"] for i in idxs}
        why = ("off-script" if whys == {"off"} else "earlier take" if whys == {"retake"}
               else "off-script and earlier take" if kind == "cut" else "off-script or earlier take")
        start, end = float(tokens[a]["start"]), float(tokens[b]["end"])
        if kind == "flag":
            why += "?"
            start = (float(tokens[a - 1]["end"]) + start) / 2 if a else 0.0
            end = (end + float(tokens[b + 1]["start"])) / 2 if b + 1 < len(tokens) else end + 1.0
        points.append({"kind": kind, "why": why, "start": round(start, 3), "end": round(max(start, end), 3),
                       "text": sep.join(tokens[i]["word"].strip() for i in idxs)})
    return points


def _line_check_summary(points: List[Dict]) -> str:
    """一句话说明按句判定做了什么，给页面和剪辑清单用。"""
    cut = sum(1 for p in points if p["kind"] == "cut")
    flagged = len(points) - cut
    if not points:
        return ""

    def _count(n: int) -> str:
        return "%d %s" % (n, "passage" if n == 1 else "passages")

    parts = []
    if cut:
        parts.append("cut %s that %s off-script or an earlier take of something said again later"
                     % (_count(cut), "is" if cut == 1 else "are"))
    if flagged:
        parts.append("kept %s that may be off-script or an earlier take, for you to check" % _count(flagged))
    return ("Line check: %s. They are listed at the top of the cut list and marked in the Premiere sequence."
            % ", and ".join(parts))


def write_cut_list(tokens: List[Dict], keep: List[Tuple[float, float]], path: str, language: str = "en",
                   notes: Tuple[str, ...] = (), points: Tuple[Dict, ...] = ()) -> None:
    """写一份可通读的剪辑清单，方便检查删了什么，再到 Premiere 里修正。

    开头先列出最值得看的地方（points，见 _check_points）：整行剪掉的和存疑保留的。
    然后是全文，按句分行：第一轮删的词标成 [-这样-]，按句判定整行剪掉的标成 [=这样=]，存疑保留的标成 [?这样?]，
    复核时删的（漏掉的口误、接缝处漏进来的残音）标成 [--这样--]，第一轮要删、复核恢复的标成 [+这样+]。
    每行开头是它在成片（序列）里的时间和在原始素材里的时间。notes 是写在说明后面的几句总结。
    """
    timeline = _EditTimeline(keep)

    def _mark(i: int) -> Optional[str]:
        cut = tokens[i].get("cut")
        if cut:
            return "[-%s-]" if cut == "first" else "[=%s=]" if cut in _TAKE_CUTS else "[--%s--]"
        return "[+%s+]" if tokens[i].get("restored") else "[?%s?]" if tokens[i].get("flag") else None

    added = sum(1 for t in tokens if t.get("cut") and t["cut"] != "first" and t["cut"] not in _TAKE_CUTS)
    restored = sum(1 for t in tokens if t.get("restored") and not t.get("cut"))
    sep = _word_sep(language)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Cut list. Columns: time in the edited sequence, time in the source, text.\n"
                "  [-like this-]    deleted by the first pass: mis-speaks, restarts, repeats and filler sounds\n"
                "  [=like this=]    whole lines cut by the line check: off-script remarks, or an earlier take of "
                "something said again later\n"
                "  [?like this?]    kept, but the line check suspects it is off-script or an earlier take\n"
                "  [--like this--]  deleted by the review (%d words): mis-speaks the first pass missed, and leftover "
                "sounds heard at the cuts\n"
                "  [+like this+]    deleted by the first pass, kept by the review (%d words)\n" % (added, restored))
        for note in notes:
            if note:
                f.write(note + "\n")
        if points:
            f.write("\nCheck first (%d places):\n" % len(points))
            for p in points:
                f.write("  %s  %s  %-34s %s\n" % (_timecode(timeline.to_edit(p["start"])), _timecode(p["start"]),
                                                  ("cut: " if p["kind"] == "cut" else "kept: ") + p["why"], p["text"]))
        f.write("\nFull transcript:\n")
        for idxs in _utterances(tokens, 0, len(tokens)):
            start = float(tokens[idxs[0]]["start"])
            f.write("%s  %s  %s\n" % (_timecode(timeline.to_edit(start)), _timecode(start),
                                      _marked_text(tokens, idxs, sep, _mark)))


def save_edit(out_path: str, audio_path: str, tokens: List[Dict], keep: List[Tuple[float, float]],
              total_dur: float, language: str, points: List[Dict]) -> str:
    """把这次的逐词判定和保留区间存到 EDIT_DIR（文件名跟输出文件走），返回路径。

    evaluate.py 用它和人工修正后的 Premiere 工程对照，算出漏删和误删。任务结束后 token 就没有了，
    不存下来就只能从调试日志里拼。
    """
    os.makedirs(EDIT_DIR, exist_ok=True)
    path = os.path.join(EDIT_DIR, os.path.splitext(os.path.basename(out_path))[0] + ".json")
    rows = []
    for t in tokens:
        row = {"word": t["word"], "start": round(float(t["start"]), 3), "end": round(float(t["end"]), 3)}
        row.update((k, t[k]) for k in ("cut", "flag", "restored", "heard") if t.get(k))
        rows.append(row)
    data = {"version": 1, "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": os.path.abspath(audio_path), "output": os.path.abspath(out_path), "language": language,
            "total_sec": round(total_dur, 3), "keep": [[round(s, 3), round(e, 3)] for s, e in keep],
            "check": points, "tokens": rows}
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(path + ".part", path)
    return path


def dedup_audio(audio_path: str, out_wav: str, api_url: str, model: str, api_key: str,
                whisper_model: str = "large", language: str = "en", keep_gap: float = 0.8,
                out_format: str = "auto", out_bitrate: str = "320k",
                progress=None, api_mode: str = "auto", review_passes: int = REVIEW_PASSES,
                take_mode: str = "cut", *, work_dir: str) -> Dict:
    """去重：只保留有文字的段落(词所在区域)，再减去语气词和 LLM 判重标记的词；无文字(静音)自然被切掉。
    接着按句判定（llm_judge_lines）：找出整行的画外话和后面重说过的前一版。take_mode 为 "cut" 时剪掉认定的行，
    "mark" 时只做标记不剪，"off" 时跳过这一步。
    然后听成片（review_edit）：把剪好的结果重新转写 review_passes 遍，修掉只有听才能发现的问题。

    返回结果里的 warnings 列出没有完全按要求完成的部分（如 LLM 窗口失败），需要展示给用户。
    """
    if take_mode not in TAKE_MODES:
        raise ValueError("take_mode must be one of %s" % ", ".join(TAKE_MODES))
    language = _norm_language(language)
    total_dur = _audio_duration(audio_path)
    warnings: List[str] = []
    _dbg("==== process start ====")
    _dbg("source total duration = %.2f s (ffprobe)" % total_dur)

    if not (api_url and model and api_key):
        raise RuntimeError("LLM dedup needs an API URL, a model name and an API key.")
    # 转写可能要跑很久：先确认 API 配置可用，配错了立刻报错，不白等
    _report(progress, 0.02, "Checking the LLM API…")
    _dbg("LLM endpoint=%s model=%s api_mode=%s" % (api_url, model, api_mode))
    llm = _LLMClient(api_url, model, api_key, api_mode)
    try:
        llm.check()
    except Exception as e:
        raise RuntimeError("LLM API check failed, please check the URL, model name and key: %s" % str(e)[:300])
    llm_mode, llm_options = llm.working
    _dbg("LLM settings in use: %s %s" % (llm_mode, llm_options))
    if not {"reasoning", "reasoning_effort"} & set(llm_options):
        warnings.append("The API did not accept a reasoning setting, so the model runs at its default. "
                        "This can make the cuts choppier.")

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
    auto: Set[int] = {i for i, w in enumerate(words)
                      if _is_vocal_sound(w["word"], words[i - 1]["word"] if i else "", core)}
    _report(progress, 0.45, "LLM dedup: analyzing words window by window…")
    llm_indices, failures, total_windows = llm_dedup_words(words, llm, progress=progress, language=language)
    # 全部失败时成片只切了静音，没有意义，不要当成结果交付
    if len(failures) == total_windows:
        raise RuntimeError("LLM dedup failed for all %d windows. First error: %s" % (total_windows, failures[0][1]))
    if failures:
        warnings.append("LLM dedup failed for %d of %d windows, so mis-speaks may remain in these parts of the source: %s. First error: %s"
                        % (len(failures), total_windows, ", ".join(where for where, _ in failures), failures[0][1]))
    proposed = auto | set(llm_indices)
    _dbg("first pass: LLM word indices(%d)+auto(%d) -> %d deleted" % (len(llm_indices), len(auto), len(proposed)))

    # 按句判定：第一轮只认得出口误，画外话和换了说法的重说要整句地读才看得出来
    takes: Dict[int, str] = {}
    flags: Dict[int, str] = {}
    if take_mode != "off":
        _report(progress, 0.51, "Line check: looking for off-script lines and earlier takes…")
        takes, flags, take_failures, take_windows = llm_judge_lines(words, proposed, llm, progress=progress,
                                                                    language=language)
        if take_failures:
            warnings.append("The line check failed for %d of %d windows, so off-script lines and earlier takes in "
                            "these parts of the source may remain or are only marked: %s. First error: %s"
                            % (len(take_failures), take_windows, ", ".join(sorted({where for where, _ in take_failures})),
                               take_failures[0][1]))
        if take_mode == "mark":
            takes, flags = {}, {**flags, **takes}

    # 第二轮：听成片。切点从这里开始就校准到真实音频的空隙上，听到的和最后交付的是同一种切法
    _report(progress, 0.55, "Measuring loudness to place the cut points…")
    env = _loudness_envelope(audio_path)
    tokens, totals, review_warnings = review_edit(
        audio_path, _edit_tokens(words, proposed, auto, takes, flags), llm, whisper_model=whisper_model,
        language=language, keep_gap=keep_gap, env=env, total_dur=total_dur, passes=max(1, int(review_passes)),
        work_dir=work_dir, progress=progress, span=(0.56, 0.90))
    warnings.extend(review_warnings)
    review_summary = _review_summary(totals)
    del_set = _cut_indices(tokens)
    _dbg("after review: %s -> delete ranges = %s"
         % (dict(totals), [(round(tokens[i]["start"], 2), round(tokens[i]["end"], 2)) for i in sorted(del_set)]))

    # 直接从「保留词」构建保留块；keep_gap 用前端设置控制切分强度
    _report(progress, 0.9, "Refining the cut points on the audio…")
    keep = clean_keep_blocks(tokens, del_set, keep_gap=keep_gap, pad=0.06, env=env)
    _dbg("keep blocks after LLM = %d: %s" % (len(keep), [(round(s,2), round(e,2)) for s,e in keep[:40]]))
    if not keep:
        raise RuntimeError("Nothing left after filtering.")
    points = _check_points(tokens, language)
    line_summary = _line_check_summary(points)
    out_cuts = os.path.splitext(out_wav)[0] + "_cuts.txt"
    try:
        write_cut_list(tokens, keep, out_cuts, language, (line_summary, review_summary), points)
    except OSError as e:  # 只是辅助文件，写不出来不值得丢掉前面的结果
        out_cuts = None
        warnings.append("The cut list could not be written: %s" % str(e)[:200])
    out_edit = None
    try:
        out_edit = save_edit(out_wav, audio_path, tokens, keep, total_dur, language, points)
    except OSError as e:  # 只用于事后评测，同样不值得丢掉结果
        _dbg("[warn] could not save the edit record: %r" % str(e)[:200])
    _report(progress, 0.90, f"Splicing ({out_format})…")
    _splice_keep(audio_path, keep, out_wav, out_format=out_format, out_bitrate=out_bitrate,
                 work_dir=work_dir, progress=progress)
    _report(progress, 0.96, "Creating the Premiere project (Premiere opens briefly)…")
    project_path = None
    try:
        project_path, project_warning = premiere.export_project(
            audio_path, keep, os.path.splitext(out_wav)[0], work_dir, log=_dbg, checks=points)
        if project_warning:
            warnings.append(project_warning)
        # 上传的文件在任务结束时删除，工程里的素材要重新链接；本地路径模式不受影响
        if _is_upload(audio_path, work_dir):
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
        "out_cuts": out_cuts,
        "out_edit": out_edit,
        "lines": line_summary,
        "review": review_summary,
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
    ap.add_argument("--review-passes", type=int, choices=[1, 2, 3], default=REVIEW_PASSES)
    ap.add_argument("--take-mode", choices=TAKE_MODES, default="cut",
                    help="off-script lines and earlier takes: cut them, only mark them, or skip the line check")
    a = ap.parse_args()
    # 输出容器跟随 --out 的扩展名，否则 AAC 源配 b.wav 会写出内容与扩展名不符的文件
    out_ext = os.path.splitext(a.out)[1].lower().lstrip(".")
    job_dir = new_job_dir()
    try:
        info = dedup_audio(a.audio, a.out, a.api_url, a.model, a.api_key, a.whisper,
                           language=a.language, api_mode=a.api_mode, review_passes=a.review_passes,
                           take_mode=a.take_mode,
                           out_format=out_ext if out_ext in ("wav", "flac", "mp3") else "auto",
                           work_dir=job_dir)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    print(json.dumps(info, indent=2))
