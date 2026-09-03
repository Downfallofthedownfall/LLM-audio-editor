#!/usr/bin/env python3
# 在全局 Python 3.14（有 CUDA torch）运行，用 CrisperWhisper verbatim 转写。
# 先 ffmpeg 解码为 16k mono WAV，再分块（30s 窗 + 重叠）逐段转写，实时把进度打到 stdout
# （供 dedup.py 流式推给 UI），避免长音频像卡住。
# 用法: py -3.14 faithful_transcribe.py <audio> <out.json> [model_size] [chunk_sec] [stride_sec]
import sys, os, json, time, subprocess, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf-cache"))
# 模型已缓存：强制离线加载，避免 transformers 联网校验缓存导致网络重置失败
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

audio = sys.argv[1]
out = sys.argv[2]
model_size = sys.argv[3] if len(sys.argv) > 3 else "large"
CHUNK = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
STRIDE = float(sys.argv[5]) if len(sys.argv) > 5 else 27.0  # 3s 重叠（避免边界丢词）

import numpy as np
from crisperwhisper.audio import load_audio, SAMPLE_RATE


def fail(msg):
    print("ERR " + msg, flush=True)
    sys.exit(1)


# 1) 加载模型（耗时最长、约 20~30s，先显示，让进度可见）
print("STAGE LOADING", flush=True)
t0 = time.time()
try:
    from crisperwhisper import CrisperWhisperModel
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
        ["ffmpeg", "-y", "-v", "error", "-i", audio,
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", wav],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0 or not os.path.exists(wav):
        fail("ffmpeg decode failed: " + (r.stderr or "")[-300:])
except Exception as e:
    fail("ffmpeg decode error: %r" % (str(e)[:300]))

y = load_audio(wav)          # soundfile 读取，16k mono float32
total = len(y) / float(SAMPLE_RATE)
try:
    os.remove(wav)
except OSError:
    pass
print("STAGE AUDIO %.1f" % total, flush=True)

if total <= 0.5:
    fail("audio too short: %.2fs" % total)

# 3) 分块转写：每块 30s（Whisper 单窗最长 30s），STRIDE 步进留重叠
all_words = []
n, warns = 0, 0
n_est = max(1, int(np.ceil(total / STRIDE)))
off = 0.0
while off < total - 0.001:
    s = int(round(off * SAMPLE_RATE))
    e = min(len(y), int(round((off + CHUNK) * SAMPLE_RATE)))
    if e - s < SAMPLE_RATE:
        break
    seg = y[s:e]
    try:
        res = model.transcribe(seg, sr=SAMPLE_RATE, language="en",
                               word_timestamps=True, mode="verbatim")
    except Exception as ex:
        print("WARN chunk failed@%.1f: %r" % (off, str(ex)[:200]), flush=True)
        warns += 1
        off += STRIDE
        continue
    for w in res.words:
        all_words.append((off + float(w.start), off + float(w.end), w.word))
    off += STRIDE
    n += 1
    print("PROGRESS %d %d" % (n, n_est), flush=True)

if not all_words:
    fail("no words transcribed (chunks=%d, warns=%d)" % (n, warns))

# 4) 边界去重：按时间排序，用覆盖游标跳过与上一块重叠的重复词
all_words.sort(key=lambda t: t[0])
final = []
last_end = 0.0
for ws, we, wd in all_words:
    if we <= last_end + 0.015:      # 已被上一块完全覆盖
        continue
    if ws < last_end - 0.05:        # 与上一块重叠的开头（重复）
        continue
    final.append({"word": wd, "start": round(float(ws), 3), "end": round(float(we), 3)})
    if we > last_end:
        last_end = we

json.dump(final, open(out, "w", encoding="utf-8"))
print("%sOK words=%d chunks=%d warns=%d dur=%.1f"
      % ("WARN " if warns else "", len(final), n, warns, total), flush=True)
