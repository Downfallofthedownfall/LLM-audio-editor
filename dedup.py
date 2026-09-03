"""
音频自动去重引擎：
  1) faster-whisper 转写英文（带词级/句级时间戳）
  2) 规则去重：连续重复词、口癖词（精确、无需模型）
  3) 可选 LLM(OpenAI 兼容，任意 chat.completions / responses 端点) 判定冗余/重复句，返回需删除的段落区间
  4) pydub 按区间剪切并拼接，输出干净音频
"""
import os, json, re, tempfile, traceback, glob
import sys, threading, datetime
# Windows 中文环境 stdout 默认 GBK，遇到非 GBK 字符会抛 UnicodeEncodeError；强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from typing import List, Dict, Optional, Tuple, Set

# 让模型/依赖的缓存落在工作区内（Windows 沙箱限制 C:\Users 下缓存写入）
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, ".hf-cache"))


def _setup_cuda_dll_path():
    """把 venv 内 nvidia cu12 库的 bin 目录加入 DLL 搜索路径，供 ctranslate2 加载 cublas/cudnn。"""
    if os.name != "nt":
        return
    sp = os.path.join(_HERE, ".venv", "Lib", "site-packages", "nvidia")
    for d in glob.glob(os.path.join(sp, "*", "bin")):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except Exception:
            pass


_setup_cuda_dll_path()

# 口癖/填充词（英文）
FILLERS = {
    "um", "uh", "erm", "er", "em", "mm", "hmm", "ah", "oh", "uhh", "like",
    "you know", "i mean", "kind of", "sort of", "actually", "basically", "literally",
}
# 常见重复词（用于连续重复检测的豁免，避免误删真正的强调重复）
AHM = set()
# 必然删除的无意义语气词（无歧义，直接删；其余交给 LLM 判断）
CORE_FILLERS = {"um", "uh", "erm", "er", "em", "mm", "hmm", "ah", "oh", "uhh", "eh"}


def _is_bracket_sound(w: str) -> bool:
    """形如 [throatclearing]、[UM]、[cough] 之类的声音标记。"""
    return bool(re.match(r"^\s*\[[^\]]+\]", w))


def _report(progress, frac: float, msg: str):
    if progress:
        try:
            progress(frac, msg)
        except Exception:
            pass


_LOG_FILE = os.path.join(_HERE, "output", "debug.log")
_dbg_lock = threading.Lock()


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


def transcribe(audio_path: str, model_size: str = "large", device: Optional[str] = None,
               language: str = "en", progress=None) -> List[Dict]:
    """用 CrisperWhisper verbatim 转写（全局 Python 3.14 + torch），返回词级 token 列表。

    子进程分块转写并实时把 STAGE/PROGRESS 打到 stdout，本函数用 Popen + 读线程
    解析并转给 UI，避免长音频看起来像卡住。
    """
    import subprocess, json as _json, threading
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "faithful_transcribe.py")
    tmp = os.path.join(here, "output", "_faithful_words.json")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    _report(progress, 0.08, f"Loading CrisperWhisper ({model_size}, verbatim)…")
    _dbg("faithful transcribe (streaming) via global py-3.14, model=%s" % model_size)
    # 清掉 PATH 里的 nvidia cu12 目录（避免全局 torch cu132 误用 venv 的 cuDNN 12）
    clean = os.environ.copy()
    pp = [p for p in clean.get("PATH", "").split(os.pathsep)
          if not any(k in p.lower() for k in ("nvidia", "cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"))]
    clean["PATH"] = os.pathsep.join(pp)

    proc = subprocess.Popen(
        ["py", "-3.14", script, audio_path, tmp, model_size],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=clean, bufsize=1,
    )
    # 进度锚点（占总进度 0.08→0.82）
    f0, f1, span = 0.08, 0.15, 0.60

    def _reader():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            _dbg("[transcribe] " + line)
            if line.startswith("STAGE LOADING"):
                _report(progress, f0, f"Loading CrisperWhisper ({model_size}, verbatim)…")
            elif line.startswith("STAGE LOADED"):
                _report(progress, f0 + 0.02, "Model loaded, ready to transcribe…")
            elif line.startswith("STAGE LOADING_AUDIO"):
                _report(progress, f0 + 0.03, "Reading audio…")
            elif line.startswith("STAGE AUDIO"):
                _report(progress, f0 + 0.04, "Audio parsed, starting chunked transcription…")
            elif line.startswith("PROGRESS"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        n, ntot = int(parts[1]), int(parts[2])
                        frac = f1 + span * (n / max(1, ntot))
                        _report(progress, min(0.80, frac), f"Transcribing chunks… {n}/{ntot}")
                    except ValueError:
                        pass
            elif line.startswith("WARN"):
                _dbg("[transcribe-warn] " + line)
            elif line.startswith("ERR"):
                _report(progress, 0.0, "Transcription failed")

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(timeout=2.0)
    if rc != 0:
        raise RuntimeError("CrisperWhisper failed (exit %d)." % rc)
    try:
        data = _json.load(open(tmp, encoding="utf-8"))
    except Exception as e:
        raise RuntimeError("CrisperWhisper produced no valid words: %r" % (str(e)[:200]))
    if not data:
        raise RuntimeError("CrisperWhisper returned no words.")
    tokens: List[Dict] = []
    for i, w in enumerate(data):
        tokens.append({
            "index": i, "word": w["word"],
            "start": float(w["start"]), "end": float(w["end"]),
            "seg_index": 0, "seg_text": w["word"],
        })
    _report(progress, 0.82, f"Verbatim transcription complete: {len(tokens)} words")
    return tokens


def _as_posix_clean(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def rule_based_removes(tokens: List[Dict]) -> List[Tuple[float, float]]:
    """连续重复词 + 口癖检测，返回需删除的 (start,end) 秒区间"""
    removes: List[Tuple[float, float]] = []
    n = len(tokens)
    # 连续重复词：例如 "I I I"、"the the"、句首重复
    i = 0
    while i < n:
        w = tokens[i]["word"]
        clean = _as_posix_clean(w)
        j = i + 1
        while j < n and _as_posix_clean(tokens[j]["word"]) == clean:
            j += 1
        # 若同一词连续出现 >=2 次，保留第一个，删掉后面重复的
        if j - i >= 2:
            # 只保留第一处，删除第 i+1..j-1
            keep_tokens = [tokens[i]]
            for k in range(i + 1, j):
                removes.append((tokens[k]["start"], tokens[k]["end"]))
            i = j
        else:
            i += 1
    # 口癖词（独立成词的填充词）：删掉该 token
    for t in tokens:
        clean = _as_posix_clean(t["word"])
        if clean in FILLERS and len(clean) <= 5:
            removes.append((t["start"], t["end"]))
    return _merge_intervals(removes)


def llm_dedup_segments(tokens: List[Dict], api_url: str, model: str, api_key: str,
                       timeout: int = 300, seg_gap: float = 0.3,
                       pause_units: Optional[List[Tuple[float, float]]] = None,
                       api_mode: str = "auto") -> List[Tuple[float, float]]:
    """调用 OpenAI 兼容接口（chat.completions / responses），按句/段判定需要删除的冗余重复内容，返回区间"""
    # 分段：优先用停顿单元(音频能量切分，对停顿敏感)，否则回退 whisper 分句
    if pause_units:
        segs = []
        ts = sorted(tokens, key=lambda t: t["start"])
        unit_words: Dict[int, List[Dict]] = {i: [] for i in range(len(pause_units))}
        for t in ts:
            mid = (t["start"] + t["end"]) / 2
            # 优先落在包含 mid 的单元；若落在停顿间隙，则归到最近的单元，绝不留白/丢词
            best = None
            best_d = float("inf")
            for i, (us, ue) in enumerate(pause_units):
                if us <= mid <= ue:
                    best = i
                    break
                d = min(abs(mid - us), abs(mid - ue))
                if d < best_d:
                    best_d = d
                    best = i
            if best is not None:
                unit_words[best].append(t)
        for i, (us, ue) in enumerate(pause_units):
            ws = sorted(unit_words[i], key=lambda w: w["start"])
            if ws:
                segs.append({"start": ws[0]["start"], "end": ws[-1]["end"],
                             "text": " ".join(w["word"] for w in ws)})
    else:
        segs = None
    if segs:
        seg_list = segs
    else:
        segs = {}
        order = []
        for t in tokens:
            si = t["seg_index"]
            if si not in segs:
                segs[si] = {"start": t["start"], "end": t["end"], "parts": []}
                order.append(si)
            segs[si]["start"] = min(segs[si]["start"], t["start"])
            segs[si]["end"] = max(segs[si]["end"], t["end"])
            segs[si]["parts"].append(t["word"])
        seg_list = [{"start": segs[si]["start"], "end": segs[si]["end"],
                     "text": " ".join(segs[si]["parts"]).strip()} for si in order]
    seg_lines = [f"[{i}] {s['start']:.2f}-{s['end']:.2f}s  {s['text']}" for i, s in enumerate(seg_list)]
    transcript = "\n".join(seg_lines)
    prompt = (
        "You are an audio-editing assistant for English speech. Below is a transcript split into "
        "numbered segments, each with a start/end time. Your job: find segments a human editor would "
        "DELETE because they are broken or redundant.\n\n"
        "DELETE these categories:\n"
        "1) FRAGMENTS / INCOMPLETE RESTARTS: a segment that is NOT a coherent complete sentence — a "
        "half-finished restart, self-correction, or broken partial phrase (e.g. \"inspired by new york "
        "city inspired by new york city i\" is a garbled restart). Delete these even if they are not "
        "exact duplicates and even when the surrounding context differs.\n"
        "2) DUPLICATE LINES: when the SAME full sentence/line is spoken more than once, keep ONLY the "
        "LATEST occurrence (the one appearing last in time) and delete ALL the earlier ones.\n"
        "3) Redundant filler or rambling that adds no new information.\n\n"
        "Do NOT delete coherent complete sentences that add new information, even if they share common "
        "phrases or mention the same topic. Be conservative: only delete segments you are confident are "
        "broken fragments or genuine duplicates.\n\n"
        "Transcript segments:\n" + transcript +
        "\n\nReturn JSON: {\"remove_segments\": [indices to DELETE]}. For duplicate lines delete all but "
        "the last; for broken fragments delete them. If none, return {\"remove_segments\": []}. JSON only."
    )
    try:
        _dbg("==== LLM dedup start ====")
        _dbg("Segments sent to LLM: %d (showing first 40):\n%s"
             % (len(segs), "\n".join(seg_lines[:40])))
        from openai import OpenAI
        client = OpenAI(base_url=api_url, api_key=api_key, timeout=timeout)
        content = _completion_text(client, model, [{"role": "user", "content": prompt}],
                                   api_mode=api_mode, json_mode=True)
        if not content:
            _dbg("[warn] LLM dedup empty reply, fallback to rule-based")
            return []
        _dbg("LLM raw reply:\n" + content)
        data = json.loads(_strip_json_fences(content))
        idxs = data.get("remove_segments", [])
        _dbg("LLM parsed remove_segments: " + repr(idxs) + " (of %d segs)" % len(seg_list))
        result = []
        for si in idxs:
            if isinstance(si, int) and 0 <= si < len(seg_list):
                result.append((seg_list[si]["start"], seg_list[si]["end"]))
        _dbg("LLM dedup -> ranges to delete: " + repr(result))
        return _merge_intervals(result)
    except Exception as e:
        _dbg("[warn] LLM dedup failed, fallback to rule-based: %r" % (str(e)[:300]))
        _dbg("[debug] segment count: %d" % len(seg_list))
        return []


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


def _subtract(blocks: List[Tuple[float, float]], removes: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """从保留片段里去掉与 removes 重叠的部分（LLM 判重用）。结果合并去重。"""
    removes = sorted(removes)
    keeps: List[Tuple[float, float]] = []
    for bs, be in blocks:
        cur = bs
        for rs, re in removes:
            # 关键修复：删除区间起点已越过本块终点，则该块剩余部分全部保留，停止处理
            if rs >= be:
                break
            rs, re = max(rs, cur), min(re, be)
            if re > cur:
                if rs > cur:
                    keeps.append((cur, rs))
                cur = max(cur, re)
        if cur < be:
            keeps.append((cur, be))
    return _merge_intervals(keeps)


def pause_segments(audio_path: str, silence_dur: float = 0.25, noise_db: float = -40.0) -> List[Tuple[float, float]]:
    """用音频能量(ffmpeg silencedetect)按停顿切分，返回说话片段 [(start_s,end_s),...]。"""
    import subprocess, re, shutil
    if shutil.which("ffmpeg") is None:
        return []
    # 检测静音区间
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af",
             f"silencedetect=noise={noise_db}dB:d={silence_dur}", "-f", "null", "-"],
            capture_output=True, text=True)
        txt = r.stderr or ""
    except Exception:
        return []
    s_starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", txt)]
    s_ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", txt)]
    # 配对成静音区间
    sil = []
    for i, st in enumerate(s_starts):
        en = s_ends[i] if i < len(s_ends) else None
        sil.append((st, en if en is not None else st + silence_dur))
    # 说话片段 = 去掉静音后的剩余
    total = _audio_duration(audio_path)
    speech = []
    cur = 0.0
    for st, en in sil:
        st = max(0.0, min(st, total))
        en = max(0.0, min(en, total))
        if st > cur:
            speech.append((cur, st))
        cur = max(cur, en)
    if cur < total:
        speech.append((cur, total))
    return [s for s in speech if s[1] - s[0] > 0.05]


def find_repeated_phrases(tokens: List[Dict], min_len: int = 6, max_len: int = 14) -> List[Tuple[float, float]]:
    """n-gram 检测全文重复的短语：对每个重复短语，保留时间上最新的一次，删掉前面所有。"""
    n = len(tokens)
    if n < min_len:
        return []
    norm = [re.sub(r"[^a-z0-9']", "", t["word"].lower()) for t in tokens]
    removes: List[Tuple[float, float]] = []
    for L in range(min_len, max_len + 1):
        occ: Dict[tuple, List[int]] = {}
        for i in range(n - L + 1):
            key = tuple(norm[i:i + L])
            if any(not w for w in key):
                continue
            occ.setdefault(key, []).append(i)
        for key, idxs in occ.items():
            if len(idxs) < 2:
                continue
            idxs.sort()
            clusters = []
            cur = [idxs[0]]
            for j in idxs[1:]:
                if j - cur[-1] <= L:
                    cur.append(j)
                else:
                    clusters.append(cur)
                    cur = [j]
            clusters.append(cur)
            if len(clusters) < 2:
                continue
            for cl in clusters[:-1]:
                removes.append((tokens[cl[0]]["start"], tokens[cl[0] + L - 1]["end"]))
    return _merge_intervals(removes)


def _ffprobe(path: str) -> Dict:
    """用 ffprobe 读取音频流参数（codec/sample_rate/channels/bit_rate）及文件时长。"""
    import subprocess, json, shutil
    if shutil.which("ffprobe") is None:
        return {}
    out = {}
    try:
        out = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
             "-of", "json", path]))
        s = (out.get("streams") or [{}])[0]
        meta = {"codec_name": s.get("codec_name", "").lower(),
                "sample_rate": s.get("sample_rate") or "44100",
                "channels": s.get("channels") or "2",
                "bit_rate": s.get("bit_rate") or ""}
    except Exception:
        meta = {"codec_name": "", "sample_rate": "44100", "channels": "2", "bit_rate": ""}
    try:
        fmt = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path]))
        meta["duration"] = fmt.get("format", {}).get("duration")
    except Exception:
        meta["duration"] = None
    return meta


def _audio_duration(path: str) -> float:
    """优先用 ffprobe 的时长（长文件/视频更可靠），失败则回退 pydub。"""
    d = _ffprobe(path).get("duration")
    if d:
        try:
            return float(d)
        except Exception:
            pass
    from pydub import AudioSegment
    return len(AudioSegment.from_file(path)) / 1000.0


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


def _splice_keep(audio_path: str, keep_blocks: List[Tuple[float, float]], out_wav: str,
                 out_format: str = "auto", out_bitrate: str = "256k") -> None:
    """用 pydub 只保留说话片段并拼接，保留原始采样率/声道，编码匹配源。"""
    from pydub import AudioSegment

    meta = _ffprobe(audio_path)
    src_codec = meta.get("codec_name", "")
    src_bit = meta.get("bit_rate", "") or ""
    fmt, codec, bitrate, lossless = _resolve_output(out_format, src_codec, src_bit, out_bitrate)

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    sr = audio.frame_rate
    ch = audio.channels
    total_dur = total_ms / 1000.0
    parts = []
    for s, e in keep_blocks:
        s_ms = int(max(0, min(float(s), total_dur)) * 1000)
        e_ms = int(max(0, min(float(e), total_dur)) * 1000)
        if e_ms - s_ms > 40:
            parts.append(audio[s_ms:e_ms])
    if not parts:
        raise RuntimeError("No speech content found.")
    _dbg(f"splice: blocks={len(parts)}, audio_dur={total_dur:.2f}s, blocks(first40)={[(round(s,2),round(e,2)) for s,e in keep_blocks][:40]}")
    out = parts[0]
    for p in parts[1:]:
        out = out + p

    params = ["-ar", str(sr), "-ac", str(ch)]
    if fmt == "mp3":
        out.export(out_wav, format="mp3", bitrate=bitrate or "256k", parameters=params)
    elif fmt == "flac":
        out.export(out_wav, format="flac", parameters=params)
    elif fmt == "mp4":  # aac/m4a
        out.export(out_wav, format="mp4", bitrate=bitrate or "256k",
                   parameters=params + ["-strict", "experimental"])
    else:
        out.export(out_wav, format="wav", parameters=params)
    out_dur = _ffprobe(out_wav).get("duration")
    _dbg(f"output duration = {out_dur} s")
    print(f"[ok] saved clean audio: {out_wav} ({sr}Hz, {ch}ch, {fmt} {bitrate or 'lossless'})")


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
    "Return ONLY a JSON array of the GLOBAL_INDEX numbers to DELETE, e.g. [0,1,2,5,10,11,12,15]. "
    "If nothing to delete, return []. JSON only — no explanation."
)


# 系统提示：让模型直接、简洁地返回 JSON，避免长篇推理/解释浪费 token
DEDUP_SYSTEM = (
    "You are a precise audio-editing assistant. You receive a transcript of spoken words, each on its own line as "
    "'GLOBAL_INDEX  WORD'. Your ONLY job is to decide which words are MIS-SPEAKS and must be DELETED, then return them "
    "as a compact JSON array of indices. Work directly and tersely: do NOT reason step-by-step, do NOT explain, do NOT "
    "quote or repeat the transcript, and do NOT output anything except the final JSON. Reply immediately with the JSON "
    "only — no thinking block, no commentary."
)


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
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


# 控制 Responses API 的思考强度：none=关闭思考；minimal/low=低强度；medium/high=高强度
# 关思考(none)会导致模型逐个词乱删（碎片化、choppy），故用 low（低强度，仍快且删得连贯）
REASONING_EFFORT = "low"


def _completion_text(client, model: str, messages: List[Dict], api_mode: str = "auto",
                     json_mode: bool = False) -> str:
    """向任意 OpenAI 兼容端点发送消息并返回助手文本。

    api_mode：
      "chat"       -> 只用 chat.completions
      "responses"  -> 只用 Responses API
      "auto"       -> 先试 Responses API，失败再回退 chat.completions
    json_mode=True 时会请求结构化 JSON 输出；若提供方拒绝该参数则自动回退到纯文本，
    调用方再负责从文本中解析 JSON。

    对 Responses API 额外发送 reasoning.effort（默认 none = 关闭思考模式），避免
    deepseek-v4 系列默认开启高强度思考导致回答被路由到 reasoning_content（content 为空）
    或又慢又费 token。
    """
    modes = {"chat": ["chat"], "responses": ["responses"],
             "auto": ["responses", "chat"]}.get(api_mode, ["responses", "chat"])
    json_flags = [True, False] if json_mode else [False]
    for mode in modes:
        for want_json in json_flags:
            try:
                if mode == "chat":
                    kwargs = dict(model=model, messages=messages, temperature=0.0)
                    if want_json:
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = client.chat.completions.create(**kwargs)
                    txt = (getattr(resp.choices[0].message, "content", "") or "").strip()
                else:
                    kwargs = dict(model=model, input=messages)
                    kwargs["reasoning"] = {"effort": REASONING_EFFORT}
                    if want_json:
                        kwargs["text"] = {"format": {"type": "json_object"}}
                    resp = client.responses.create(**kwargs)
                    txt = _resp_text(resp).strip()
                if txt:
                    return txt
            except Exception as e:
                _dbg("[warn] llm %s json=%s failed: %r" % (mode, want_json, str(e)[:200]))
    return ""


def llm_dedup_words(words: List[Dict], api_url: str, model: str, api_key: str,
                    timeout: int = 600, progress=None, api_mode: str = "auto") -> List[int]:
    """调用 OpenAI 兼容接口（chat.completions / responses）分窗判重。

    每窗约 WIN 词 + LOOK 回看上下文，多窗并行。可在 api_mode 中指定使用的端点类型
    （auto 会自动在 chat.completions 与 responses 之间回退）。返回要删除的全局词索引数组。"""
    n = len(words)
    if n == 0:
        return []
    WIN, LOOK = 500, 70
    windows = []
    a = 0
    while a < n:
        b = min(n, a + WIN)
        ctx = max(0, a - LOOK) if a > 0 else 0
        windows.append((ctx, b, a))
        if b >= n:
            break
        a = b
    total = len(windows)
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_url, api_key=api_key, timeout=timeout)
    except Exception as e:
        _dbg("[warn] OpenAI client init failed: %r" % (str(e)[:200]))
        return []

    def _window(w):
        ctx, b, acpt = w
        lines = _win_lines(words, ctx, b)
        if ctx < acpt:
            lines = ["... (above is the previous window's context, for reference only; may also be judged a mistake) ..."] + lines
        readable = "\n".join(lines)
        _dbg("==== window (accepted %d-%d) ====\n%s" % (acpt, b, "\n".join(lines[:25])))
        user_prompt = (DEDUP_RULES + "\n\n---\n\nTRANSCRIPT (each word on its own line; the LEFT number is that "
                       "word's GLOBAL index):\n" + readable +
                       "\n\nReturn ONLY a JSON object {\"indices\":[...]} listing the GLOBAL index numbers to DELETE.")
        messages = [{"role": "system", "content": DEDUP_SYSTEM},
                    {"role": "user", "content": user_prompt}]
        content = _completion_text(client, model, messages, api_mode=api_mode, json_mode=True)
        if not content:
            _dbg("[warn] window %d-%d returned nothing" % (acpt, b))
            return set()
        _dbg("window %d-%d output len=%d tokens~%d"
             % (acpt, b, len(content), len(content.split())))
        try:
            parsed = json.loads(_strip_json_fences(content))
        except Exception as e:
            _dbg("[warn] window %d-%d could not parse JSON: %r" % (acpt, b, str(e)[:150]))
            return set()
        idxs = parsed.get("indices", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(idxs, list):
            idxs = []
        got = sorted({int(x) for x in idxs
                      if isinstance(x, (int, float)) and acpt <= int(x) < b})
        _dbg("window %d-%d -> added %d" % (acpt, b, len(got)))
        return set(got)

    deleted: set = set()
    if total == 1:
        deleted = _window(windows[0])
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(2, total)) as ex:
            futs = [ex.submit(_window, w) for w in windows]
            done = 0
            for fut in as_completed(futs):
                try:
                    deleted |= fut.result()
                except Exception as e:
                    _dbg("[warn] window task raised: %r" % (str(e)[:150]))
                done += 1
                if progress:
                    try:
                        progress(0.72 + 0.12 * done / total, "LLM dedup %d/%d windows…" % (done, total))
                    except Exception:
                        pass
    _dbg("LLM dedup total deleted = %d" % len(deleted))
    return sorted(deleted)


def detect_silence_segments(words: List[Dict], threshold: float = 0.5) -> List[Tuple[float, float]]:
    """检测词间静音（gap >= threshold 秒），返回要删的静音区间 [(start,end)]。"""
    segs = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= threshold:
            segs.append((words[i - 1]["end"], words[i]["start"]))
    return _merge_intervals(segs)


def build_delete_segments(words: List[Dict], indices: List[int], merge_gap: float = 0.05) -> List[Tuple[float, float]]:
    """把词序号映射为时间区间，并按 merge_gap 合并相邻区间（拷贝 videocut 逻辑）。"""
    if not indices:
        return []
    ranges = [(words[i]["start"], words[i]["end"]) for i in indices if 0 <= i < len(words)]
    if not ranges:
        return []
    ranges.sort(key=lambda r: r[0])
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _tc(sec: float, fps: int = 25) -> str:
    """秒 -> HH:MM:SS:FF 时间码。"""
    frames = int(round(sec * fps))
    f = frames % fps
    total_s = frames // fps
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return "%02d:%02d:%02d:%02d" % (h, m, s, f)


def export_aaf(audio_path: str, keep_blocks: List[Tuple[float, float]], out_aaf: str,
               sr: int = 48000, edit_rate: int = 25, name: str = "Dedup",
               total_dur: Optional[float] = None) -> str:
    """把保留片段导出为 AAF 序列（引用外部源文件，Premiere 可链接一个源并替换素材）。"""
    import aaf2, urllib.parse
    blocks = sorted((s, e) for s, e in keep_blocks if e - s > 0.02)
    if not blocks:
        raise RuntimeError("No keep blocks for aaf.")
    if total_dur is None:
        total_dur = _audio_duration(audio_path)
    url = "file:///" + urllib.parse.quote(os.path.abspath(audio_path).replace("\\", "/"))
    src_name = os.path.basename(audio_path)
    total_samples = int(total_dur * sr)
    with aaf2.open(out_aaf, "w") as f:
        file_mob = f.create.SourceMob(src_name)
        loc = f.create.NetworkLocator()
        loc['URLString'].value = url
        wav = f.create.WAVEDescriptor()
        wav['SampleRate'].value = "%d/1000" % sr
        wav['Length'].value = total_samples
        wav['Summary'].value = b'\x00\x00\x00\x00'
        wav.locator.append(loc)
        file_mob.descriptor = wav
        _, slot = file_mob.create_essence(edit_rate=edit_rate, media_kind='sound', offline=True)
        f.content.mobs.append(file_mob)
        master = f.create.MasterMob("MASTER")
        f.content.mobs.append(master)
        clip = file_mob.create_source_clip(slot_id=slot.slot_id, media_kind='sound')
        mslot = master.create_timeline_slot(edit_rate=edit_rate)
        mslot.segment = clip
        comp = f.create.CompositionMob(name)
        f.content.mobs.append(comp)
        seq_slot = comp.create_empty_sequence_slot(edit_rate, media_kind='sound')
        seq = seq_slot.segment
        for s, e in blocks:
            c = master.create_source_clip(slot_id=mslot.slot_id, media_kind='sound')
            c.start = int(round(s * edit_rate))
            c.length = int(round((e - s) * edit_rate))
            seq.components.append(c)
    _dbg("AAF written: %s (%d clips, ref=%s)" % (out_aaf, len(blocks), url))
    return out_aaf


def export_edl(audio_path: str, keep_blocks: List[Tuple[float, float]], out_edl: str,
               fps: int = 25, title: str = "Step-Audio EditX Dedup") -> str:
    """把保留片段导出为 Premiere 可导入的 CMX3600 EDL（剪辑单）。"""
    blocks = sorted((s, e) for s, e in keep_blocks if e - s > 0.02)
    if not blocks:
        raise RuntimeError("No keep blocks for edl.")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "", os.path.splitext(os.path.basename(audio_path))[0])
    reel = (safe_name[:8] or "SRC")
    lines = ["TITLE: " + title, "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    for i, (s, e) in enumerate(blocks):
        si, so = _tc(s, fps), _tc(e, fps)
        ri, ro = _tc(rec, fps), _tc(rec + (e - s), fps)
        lines.append("%03d  %-8s V  C  %s %s %s %s" % (i + 1, reel, si, so, ri, ro))
        rec += (e - s)
    lines.append("")
    with open(out_edl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _dbg("EDL written: %s (%d clips, fps=%d)" % (out_edl, len(blocks), fps))
    return out_edl


def export_fcpxml(audio_path: str, keep_blocks: List[Tuple[float, float]], out_fcpxml: str,
                  total_dur: Optional[float] = None, name: str = "Step-Audio EditX Dedup") -> str:
    """把保留片段导出为 Premiere 可导入的 FCPXML 序列（时间线上每个保留段是一个 clip）。"""
    import urllib.parse
    if total_dur is None:
        total_dur = _audio_duration(audio_path)
    blocks = sorted((s, e) for s, e in keep_blocks if e - s > 0.02)
    if not blocks:
        raise RuntimeError("No keep blocks for fcpxml.")
    abs_path = os.path.abspath(audio_path).replace("\\", "/")
    uri = "file:///" + urllib.parse.quote(abs_path)
    base = os.path.basename(abs_path)
    clips = []
    for i, (s, e) in enumerate(blocks):
        clips.append(f'      <asset-clip ref="a1" start="{s:.3f}s" duration="{e - s:.3f}s" name="seg{i}"/>')
    spine = "\n".join(clips)
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<fcpxml version="1.9">\n'
        '  <resources>\n'
        '    <format id="r1" name="FFVideoFormatRateUndefined" frameDuration="600/18000"/>\n'
        f'    <asset id="a1" name="{base}" start="0s" duration="{total_dur:.3f}s" hasVideo="1" hasAudio="1">\n'
        f'      <media-rep kind="original-media" src="{uri}"/>\n'
        '    </asset>\n'
        '  </resources>\n'
        '  <library>\n'
        f'    <event name="{name}">\n'
        '      <project name="Dedup Sequence">\n'
        '        <sequence format="r1">\n'
        '          <spine>\n'
        f'{spine}\n'
        '          </spine>\n'
        '        </sequence>\n'
        '      </project>\n'
        '    </event>\n'
        '  </library>\n'
        '</fcpxml>\n'
    )
    with open(out_fcpxml, "w", encoding="utf-8") as f:
        f.write(doc)
    _dbg("FCPXML sequence written: %s (%d clips)" % (out_fcpxml, len(blocks)))
    return out_fcpxml


def clean_keep_blocks(words: List[Dict], del_set: Set[int], keep_gap: float = 0.4,
                      pad: float = 0.06) -> List[Tuple[float, float]]:
    """直接从「保留词」构建保留块。

    被删的词会在中途切断块（不跨越删除区合并），因此被删的整段——连同它周边的
    微秒残响——都不会出现在输出里，避免脏残留 / 咔哒声。保留块内部保留 <=keep_gap
    的自然停顿，并在块边缘加 pad 以便干净剪切。关键是：**pad 不会越过相邻被删词
    的边界**（clamp 到被删词的 start/end），所以被删口误的边缘残响不会被带回来
    （消除"pad 截断"声）。最后合并因 pad 而相接/重叠的块。
    """
    blocks: List[Tuple[float, float]] = []
    del_spans: List[Tuple[float, float]] = []
    cur_s = cur_e = None
    for i, w in enumerate(words):          # words 已是时间顺序
        s, e = float(w["start"]), float(w["end"])
        if i in del_set:
            # 被删的词：立即闭合当前块，让删除区彻底断开
            if cur_s is not None:
                blocks.append((cur_s, cur_e))
                cur_s = cur_e = None
            del_spans.append((s, e))
            continue
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s - cur_e <= keep_gap:
            cur_e = max(cur_e, e)
        else:
            blocks.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        blocks.append((cur_s, cur_e))
    del_spans.sort()
    import bisect
    padded: List[Tuple[float, float]] = []
    for s, e in blocks:
        if e - s <= 0.02:
            continue
        ps, pe = max(0.0, s - pad), e + pad
        # 左侧最近的被删词（end < s）：pad 起点不低于它的结尾
        i1 = bisect.bisect_left(del_spans, (s, float("-inf"))) - 1
        if i1 >= 0:
            ps = max(ps, del_spans[i1][1])
        # 右侧最近的被删词（start > e）：pad 终点不越过它的开头
        i2 = bisect.bisect_right(del_spans, (e, float("inf")))
        if i2 < len(del_spans):
            pe = min(pe, del_spans[i2][0])
        if pe > ps:
            padded.append((ps, pe))
    # 合并因 pad 而相接/重叠的相邻块，避免拼接时同一段被重复（叠音/没对齐）
    merged: List[Tuple[float, float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def dedup_audio(audio_path: str, out_wav: str, api_url: str, model: str, api_key: str,
                whisper_model: str = "base", language: str = "en", keep_gap: float = 0.8,
                use_llm: bool = False, out_format: str = "auto", out_bitrate: str = "256k",
                seg_gap: float = 0.3, pause_dur: float = 0.25, progress=None,
                api_mode: str = "auto") -> Dict:
    """去重：只保留有文字的段落(词所在区域)，再减去 LLM 判重标记的词；无文字(静音)自然被切掉。"""
    total_dur = _audio_duration(audio_path)
    _dbg("==== process start ====")
    _dbg("source total duration = %.2f s (ffprobe)" % total_dur)
    words = transcribe(audio_path, whisper_model, language=language, progress=progress)
    if not words:
        raise RuntimeError("Transcription returned no words.")
    _dbg("words = %d; first = %.2f-%.2f; last = %.2f-%.2f"
         % (len(words), words[0]["start"], words[0]["end"], words[-1]["start"], words[-1]["end"]))

    # 只有文字段落：词相邻(gap<0.4s)合并成块；无文字(静音/空白)区域自然被排除（仅用于日志参考）
    text_blocks = speech_blocks(words, keep_gap=0.4)
    _dbg("text blocks (has text) = %d: %s" % (len(text_blocks), [(round(s,2), round(e,2)) for s,e in text_blocks[:30]]))
    _dbg("text total = %.2f s (of %.2f total)" % (sum(e-s for s,e in text_blocks), total_dur))

    del_set: Set[int] = set()
    if use_llm and api_key and api_url and model:
        _report(progress, 0.7, "LLM dedup: analyzing words window by window…")
        llm_indices = llm_dedup_words(words, api_url, model, api_key, progress=progress, api_mode=api_mode)
        # 保险：无意义声音标记 + 纯语气词必然删（即使 LLM 漏判）
        auto = set()
        for i, w in enumerate(words):
            if _is_bracket_sound(w["word"]) or _as_posix_clean(w["word"]) in CORE_FILLERS:
                auto.add(i)
        del_idx = sorted(set(llm_indices) | auto)
        del_set = set(del_idx)
        _dbg("LLM word indices(%d)+auto(%d) -> delete ranges = %s"
             % (len(llm_indices), len(auto),
                [(round(words[i]['start'],2), round(words[i]['end'],2)) for i in del_idx]))

    # 直接从「保留词」构建保留块：被删的词整段消失、不留 pad 残响
    keep = clean_keep_blocks(words, del_set, keep_gap=0.4, pad=0.06)
    _dbg("keep blocks after LLM = %d: %s" % (len(keep), [(round(s,2), round(e,2)) for s,e in keep[:40]]))
    if not keep:
        raise RuntimeError("Nothing left after filtering.")
    _report(progress, 0.93, f"Splicing ({out_format})…")
    _splice_keep(audio_path, keep, out_wav, out_format=out_format, out_bitrate=out_bitrate)
    aaf_path = os.path.splitext(out_wav)[0] + ".aaf"
    try:
        export_aaf(audio_path, keep, aaf_path)
    except Exception as e:
        _dbg("[warn] aaf export failed: %r" % (str(e)[:200]))
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
        "out_aaf": aaf_path if os.path.exists(aaf_path) else None,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-mode", choices=["auto", "chat", "responses"], default="auto")
    ap.add_argument("--whisper", default="base")
    a = ap.parse_args()
    info = dedup_audio(a.audio, a.out, a.api_url, a.model, a.api_key, a.whisper, api_mode=a.api_mode)
    print(json.dumps(info, indent=2))
