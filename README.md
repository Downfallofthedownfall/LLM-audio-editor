# Audio Dedup

A local, web-based audio editor that **automatically removes mis-speaks, restarts,
stutters, duplicate phrases, and fillers** from an English narration (video or audio),
and produces a **clean audio file** plus a **Premiere Pro–ready AAF sequence** for a
manual-correction workflow.

The key insight driving the design: **the ASR must be faithful (verbatim).** Standard
Whisper normalizes disfluencies away, so the filler words and stutters never appear in
the transcript and therefore can never be removed. This tool uses a verbatim
transcriber so every "um", "uh", "[throatclearing]", and repeated phrase is preserved
and can be flagged for deletion, then the audio is re-assembled from only the kept
parts.

---

## What it does

1. **Faithful transcription** — transcribes the audio word-by-word in **verbatim mode**,
   preserving interjections, cut-offs (`I-`), and vocal sounds (`[throatclearing]`,
   `[UM]`, …), each with a start/end timestamp.
2. **Automatic silence trimming** — keeps only spoken regions and drops long silences.
3. **AI mis-speak removal (optional)** — sends the transcript in small windows to an
   LLM (DeepSeek), which marks which words are mis-speaks (restarts, repeated phrases,
   `-` cut-offs, fillers) using a **"keep the last (successful) version, delete the
   earlier repeats"** rule, and returns the indices to delete.
4. **Guaranteed cleanups** — bracketed vocal sounds and core fillers are always removed,
   independent of the LLM.
5. **Re-assembly** — keeps only the non-deleted words (deleted words break the block, so
   no residue / clicks) and concatenates them cleanly.
6. **Export** — a clean `wav` / `flac` / `mp3`, plus a **Premiere Pro AAF sequence** that
   links to the original source for human correction.

---

## Pipeline

```
audio/video ──ffmpeg──▶ 16 kHz mono ──CrisperWhisper (verbatim)──▶ words[start,end]
                                                                        │
                 speech_blocks (merge kept words, drop silences) ◀──────┘
                                                                        │
                LLM (DeepSeek) windowed, per-word index dedup ─────▶ delete indices
                                                                        │
                        clean_keep_blocks ──▶ kept blocks (no residue)
                                                                        │
                pydub splice ──▶ out.wav   +   pyaaf2 ──▶ out.aaf (Premiere)
```

- **Transcription** runs in a separate Python interpreter that has a CUDA PyTorch.
- **Everything else** (LLM, splice, AAF) runs in the web-server virtual environment.

---

## Features

- **Verbatim ASR** that keeps disfluencies, so they can actually be deleted.
- **AI dedup** with a clear per-word rule (keep the last, delete the earlier), windowed
  so it stays fast and reliable on long files.
- **No residue / no clicks**: deleted segments are removed entirely, including their
  cut-padding edges.
- **LLM speed controls**: `thinking` disabled, parallel windows, system-prompt
  conciseness, and automatic fallback so the job never silently produces an empty result.
- **Multiple outputs**: lossless `wav` / `flac`, high-bitrate `mp3`, and a **Premiere Pro
  AAF** sequence that references the original file (one source to re-link).
- **Offline model loading**: uses the locally cached Whisper model, never touches the
  network at inference time.
- **Real-time server-sent progress** in the browser while processing.

---

## Requirements

- **Windows** (paths and commands below are Windows-specific, but the logic is portable).
- **Python 3.12+** for the server `venv`.
- **A CUDA-enabled Python 3.14** environment with PyTorch for transcription (see
  Installation).
- **ffmpeg / ffprobe** on `PATH` (decoding and silence detection).
- A **DeepSeek** API key (OpenAI-compatible) with a model that supports `thinking`
  control (e.g. `deepseek-v4-flash`).

---

## Installation

### 1. Server virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs Flask, openai, pydub, pyaaf2, and numpy.

### 2. Transcription environment (CUDA PyTorch, separate interpreter)

The verbatim transcriber (`faithful_transcribe.py`) needs a PyTorch build with CUDA.
Install it in a *separate* interpreter (e.g. your global Python 3.14):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install transformers crispterwhisper[convert] librosa soundfile soxr numpy
```

> **Note:** On Windows, the CrisperWhisper **CTranslate2 backend is not available**
> (its `ctranslate2-crisperwhisper` fork ships Linux-only wheels). The tool therefore
> uses the **transformers** backend, which works but is slower on very long files.
> The pipeline mitigates this with chunked transcription and live progress.

### 3. ffmpeg / ffprobe

Ensure `ffmpeg` and `ffprobe` are on `PATH` (e.g. installed via `winget install
Gyan.FFmpeg`).

### 4. Download the model

The first run downloads the CrisperWhisper model (e.g. `CrisperWhisper2.0_large`) into
`HF_HOME`. Afterwards it is loaded **offline**.

---

## Configuration

- **Model**: the default is `large` (most accurate); `small` is available for speed.
  Both must be present in the local HF cache.
- **API**: set the OpenAI-compatible `api_url` (default `https://api.deepseek.com`),
  `model` (default `deepseek-chat`), and your API key in the web form.
- **Dedup** uses the `thinking: disabled` parameter to keep the reasoning model terse and
  fast, sends the transcript in **windows** of ~500 words, and caps output so the answer
  is never truncated.

---

## Usage

### Start the server

Double-click `start.bat` (creates the venv-backed server and opens the browser), or run:

```powershell
.\.venv\Scripts\python.exe server.py
```

Then open **http://127.0.0.1:7861**.

### In the web UI

1. Select your audio/video file.
2. Fill in the DeepSeek **API URL**, **model name**, and **API key**.
3. Choose the **Whisper model** (`large` default) and output settings.
4. Optionally enable **"Use DeepSeek to remove mis-speaks"**.
5. Click **Run**. Watch the live progress bar.
6. Download the clean audio and/or the **AAF** for Premiere Pro.

### AAF in Premiere Pro

- Import the `.aaf` file.
- It references the **original source file** (a single source). Relink it if prompted.
- The timeline shows each kept segment as an independent clip for easy manual correction.

---

## Outputs

- **`wav` / `flac`** — lossless, preserving the source sample rate and channels.
- **`mp3`** — high-bitrate (default 320k) compressed.
- **`.aaf`** — Premiere Pro sequence (all kept segments, referencing the source).

---

## Project layout

```
Root Folder
  server.py              # Flask web server + SSE progress + upload/static
  dedup.py               # dedup engine: transcription orchestration, LLM dedup,
                         #   keep-block building, splice, AAF/EDL/FCPXML export
  faithful_transcribe.py # verbatim transcription subprocess (runs in global CUDA env)
  start.bat              # one-click server launcher
  requirements.txt       # server venv dependencies
  .gitignore
```

- `server.py` runs in `.venv`.
- `dedup.py` calls `faithful_transcribe.py` (in the CUDA Python 3.14 env) as a
  subprocess, streaming progress back to the UI.

---

## Troubleshooting

- **"CrisperWhisper failed (exit 1)"** — the model load tried to reach Hugging Face and
  was reset. The tool now forces **offline** loading (`HF_HUB_OFFLINE=1`); ensure the
  model is fully cached in `HF_HOME`.
- **LLM returns nothing / "empty reply"** — the reasoning model ate the token budget.
  The tool disables `thinking` and does not set a small token limit; if a window still
  fails it automatically falls back to a simpler call. Check `output\debug.log` for the
  `window ... -> added N` lines.
- **Clicks / residue at cut points** — fixed by building keep blocks from kept words
  only, so deleted segments (including their padding) are removed entirely.
- **Slow on long files** — the transformers backend is slower than CTranslate2 on
  Windows; the tool chunk-transcribes and streams progress so it is never "stuck".
- **Debug log** — look in `output\debug.log` for transcription, LLM replies, and the
  final keep/delete ranges.

---

## License & notes

- The transcription model (CrisperWhisper) is licensed for **non-commercial research**.
  Use it for your own personal projects accordingly.
- This project is provided as-is for local, personal use.
