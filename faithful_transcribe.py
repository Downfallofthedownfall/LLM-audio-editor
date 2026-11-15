#!/usr/bin/env python3
# 在全局 Python 3.14（有 CUDA torch）运行，用 CrisperWhisper verbatim 转写。
# 先 ffmpeg 解码为 16k mono WAV，再分块（30s 窗 + 重叠）逐段转写，实时把进度打到 stdout
# （供 dedup.py 流式推给 UI），避免长音频像卡住。
# 用法: py -3.14 faithful_transcribe.py <audio> <out.json> [model_size] [chunk_sec] [stride_sec] [language]
import difflib, gc, glob, json, math, os, re, subprocess, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 单块转写的总尝试次数。重试用尽就让任务失败：这段没有词的话，会被当成静音从成片里删掉
CHUNK_ATTEMPTS = 3
# 相邻两块在重叠区听到同一个词时，两边给它的时间最多差这么多（实测一般几十毫秒，个别几百毫秒）
SEAM_MATCH_SEC = 0.5


def fail(msg):
    print("ERR " + msg, flush=True)
    sys.exit(1)


def model_is_cached(model_size):
    """本地缓存里是否已有完整模型（有快照，且没有下载到一半的文件）。"""
    hub = glob.escape(os.path.join(os.environ["HF_HOME"], "hub"))
    for repo in glob.glob(os.path.join(hub, "models--*--CrisperWhisper*_" + glob.escape(model_size))):
        repo = glob.escape(repo)
        if (glob.glob(os.path.join(repo, "snapshots", "*", "config.json"))
                and not glob.glob(os.path.join(repo, "blobs", "*.incomplete"))):
            return True
    return False


def free_gpu_memory():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe_chunk(model, seg, sample_rate, lang, off):
    """转写一块，失败时释放显存后重试，重试用尽则终止任务。返回 (结果, 失败次数)。"""
    last_error = ""
    for attempt in range(1, CHUNK_ATTEMPTS + 1):
        try:
            res = model.transcribe(seg, sr=sample_rate, language=lang,
                                   word_timestamps=True, mode="verbatim")
            return res, attempt - 1
        except Exception as ex:
            last_error = str(ex)[:200]
            print("WARN chunk failed@%.1f (attempt %d/%d): %r"
                  % (off, attempt, CHUNK_ATTEMPTS, last_error), flush=True)
        free_gpu_memory()
    fail("transcription failed at %.1fs after %d attempts: %s" % (off, CHUNK_ATTEMPTS, last_error))


def _word_key(word):
    """比较两块听到的是不是同一个词：只看文字，忽略大小写和标点。"""
    return re.sub(r"[^\w']", "", word.lower())


def _seam_anchor(tail, head, middle):
    """在重叠区里找一个两块都听到的词，返回 (tail 里的下标, head 里的下标)；没有就返回 None。

    按文字对齐，再要求两块给它的时间相差不超过 SEAM_MATCH_SEC（同一个词在重叠区里可能说了两遍）。
    有多个时取最靠近重叠区中点的：那里离两块的窗口边缘都够远，两边都听得完整。
    """
    matcher = difflib.SequenceMatcher(None, [_word_key(w[2]) for w in tail],
                                      [_word_key(w[2]) for w in head], autojunk=False)
    best = None
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            x, y = tail[a + k], head[b + k]
            mid_x, mid_y = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            if not _word_key(x[2]) or abs(mid_x - mid_y) > SEAM_MATCH_SEC:
                continue
            distance = abs((mid_x + mid_y) / 2 - middle)
            if best is None or distance < best[0]:
                best = (distance, a + k, b + k)
    return best[1:] if best else None


def stitch_chunks(chunks, chunk_sec, stride_sec):
    """把各块的词按时间拼起来。chunks 为按时间排列的 [(块起点, [(start, end, word), ...]), ...]。

    相邻两块有一段重叠，同一个词两块各听到一次，时间戳还会差几十到几百毫秒，所以不能只靠时间判断
    哪些词重复：那样时间戳一不一致就会丢词或重复。这里在重叠区里找一个两块都听到的词（_seam_anchor），
    在它身上交接：它和它之前的词用上一块的，之后的词用下一块的，每个词只出现一次。
    重叠区里找不到相同的词时（一边没听到词，或两边听得完全不同），才退回按时间在重叠区中点交接。
    """
    final = []
    for n, (off, words) in enumerate(chunks):
        words = list(words)
        if n == 0:
            final.extend(words)
            continue
        lo, hi = off, chunks[n - 1][0] + chunk_sec  # 两块重叠的时间范围
        first_tail = len(final)
        while first_tail > 0 and final[first_tail - 1][1] > lo:
            first_tail -= 1
        n_head = 0
        while n_head < len(words) and words[n_head][0] < hi:
            n_head += 1
        tail, head = final[first_tail:], words[:n_head]
        anchor = _seam_anchor(tail, head, (lo + hi) / 2) if tail and head else None
        if anchor:
            a, b = anchor
            (ws, we, wd), b_end = tail[a], head[b][1]
            del final[first_tail + a:]
            # 这个词的开头用上一块的时间、结尾用下一块的时间：它和前后两个词的分界各由同一块给出，不会互相矛盾
            final.append((ws, b_end if b_end > ws else we, wd))
            final.extend(words[b + 1:])
            continue
        if head:  # 下一块在重叠区听到了词：上一块在中点之后的词（可能被窗口截断）让给它
            cut = (lo + hi) / 2
            while len(final) > first_tail and (final[-1][0] + final[-1][1]) / 2 >= cut:
                final.pop()
        last_end = final[-1][1] if final else 0.0
        # 大半已被上一块留下的词盖住的，是同一段声音的另一种听法，不再重复
        final.extend(w for w in words if (w[0] + w[1]) / 2 > last_end)
    return [{"word": wd, "start": round(float(ws), 3), "end": round(float(we), 3)} for ws, we, wd in final]


def main():
    audio = sys.argv[1]
    out = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "large"
    chunk_sec = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    stride_sec = float(sys.argv[5]) if len(sys.argv) > 5 else 27.0  # 3s 重叠（避免边界丢词）
    lang = sys.argv[6] if len(sys.argv) > 6 else "en"
    if lang in ("auto", "", "none", "None"):
        lang = "en"

    # 模型已缓存：强制离线加载，避免 transformers 联网校验缓存导致网络重置失败。
    # 还没缓存（首次运行）时保持联网，才能下载模型。必须在导入 transformers 之前设置。
    if model_is_cached(model_size):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 1) 加载模型（耗时最长、约 20~30s，先显示，让进度可见）
    print("STAGE LOADING", flush=True)
    t0 = time.time()
    try:
        import soundfile as sf
        from crisperwhisper import CrisperWhisperModel
        from crisperwhisper.audio import SAMPLE_RATE
        model = CrisperWhisperModel(model_size)
    except Exception as e:
        fail("model load failed: %r" % (str(e)[:300]))
    print("STAGE LOADED %.1f" % (time.time() - t0), flush=True)

    # 2) ffmpeg 解码成 16k mono WAV（处理任意格式，避开 librosa 缓存 bug）
    workdir = os.path.dirname(os.path.abspath(out)) or "."
    wav = os.path.join(workdir, "_faithful_decode.wav")
    print("STAGE LOADING_AUDIO", flush=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", audio, "-map", "0:a:0",
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", wav],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0 or not os.path.exists(wav):
            fail("ffmpeg decode failed: " + (r.stderr or "")[-300:])
    except Exception as e:
        fail("ffmpeg decode error: %r" % (str(e)[:300]))

    wav_file = sf.SoundFile(wav)  # 不整体载入：转写时每次只读当前这一块到内存
    total_frames = wav_file.frames
    total = total_frames / float(SAMPLE_RATE)
    print("STAGE AUDIO %.1f" % total, flush=True)

    if total <= 0.5:
        fail("audio too short: %.2fs" % total)

    # 3) 分块转写：每块 30s（Whisper 单窗最长 30s），stride_sec 步进留重叠
    chunks = []
    warns = 0
    n_est = max(1, math.ceil(total / stride_sec))
    off = 0.0
    while off < total - 0.001:
        s = int(round(off * SAMPLE_RATE))
        e = min(total_frames, int(round((off + chunk_sec) * SAMPLE_RATE)))
        if off > 0 and e - s < SAMPLE_RATE:  # 不足 1 秒的尾巴已被上一块的重叠覆盖
            break
        wav_file.seek(s)
        seg = wav_file.read(e - s, dtype="float32", always_2d=False)
        res, failures = transcribe_chunk(model, seg, SAMPLE_RATE, lang, off)
        warns += failures
        # 一个词都没识别出（静音、纯音乐）时库返回 words=None
        chunks.append((off, [(off + float(w.start), off + float(w.end), w.word)
                             for w in (res.words or [])]))
        off += stride_sec
        print("PROGRESS %d %d" % (len(chunks), n_est), flush=True)

    wav_file.close()
    try:
        os.remove(wav)
    except OSError:
        pass

    # 4) 拼接各块、去掉重叠区的重复词
    final = stitch_chunks(chunks, chunk_sec, stride_sec)
    if not final:
        fail("no words transcribed (chunks=%d)" % len(chunks))

    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f)
    print("%sOK words=%d chunks=%d warns=%d dur=%.1f"
          % ("WARN " if warns else "", len(final), len(chunks), warns, total), flush=True)


if __name__ == "__main__":
    main()
