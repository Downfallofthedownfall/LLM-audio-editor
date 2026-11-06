#!/usr/bin/env python3
# 在全局 Python 3.14（有 CUDA torch）运行，用 CrisperWhisper verbatim 转写。
# 先 ffmpeg 解码为 16k mono WAV，再分块（30s 窗 + 重叠）逐段转写，实时把进度打到 stdout
# （供 dedup.py 流式推给 UI），避免长音频像卡住。
# 用法: py -3.14 faithful_transcribe.py <audio> <out.json> [model_size] [chunk_sec] [stride_sec] [language]
import gc, glob, json, math, os, subprocess, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 单块转写的总尝试次数。重试用尽就让任务失败：这段没有词的话，会被当成静音从成片里删掉
CHUNK_ATTEMPTS = 3


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


def stitch_chunks(chunks, chunk_sec, stride_sec):
    """把各块的词按时间拼起来。chunks 为按时间排列的 [(块起点, [(start, end, word), ...]), ...]。

    每块只用到重叠区中点为止：块尾的词可能被窗口截断，留给下一块（那里离窗口边缘够远）。
    再用覆盖游标跳过上一块已经说过的词。
    """
    half_overlap = max(0.0, chunk_sec - stride_sec) / 2
    final = []
    last_end = 0.0
    for i, (off, words) in enumerate(chunks):
        cut = off + stride_sec + half_overlap if i < len(chunks) - 1 else float("inf")
        for ws, we, wd in words:
            if (ws + we) / 2 >= cut:        # 留给下一块
                continue
            if we <= last_end + 0.015:      # 已被上一块完全覆盖
                continue
            if ws < last_end - 0.05:        # 与上一块重叠的开头（重复）
                continue
            final.append({"word": wd, "start": round(float(ws), 3), "end": round(float(we), 3)})
            if we > last_end:
                last_end = we
    return final


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
