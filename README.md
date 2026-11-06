# Audio Dedup

A local, web-based audio editor that **automatically removes mis-speaks, restarts,
stutters, duplicate phrases, and fillers** from an English narration (video or audio),
and produces a **clean audio file** plus a **Premiere Pro project** with the video and audio
already cut, for a manual-correction workflow.

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
3. **AI mis-speak removal** — sends the transcript in small windows to an
   LLM (DeepSeek), which marks which words are mis-speaks (restarts, repeated phrases,
   `-` cut-offs, fillers) using a **"keep the last (successful) version, delete the
   earlier repeats"** rule, and returns the indices to delete. A **second pass** then reads the
   text as a listener would hear it, restores words that were wrongly deleted, and deletes
   mistakes that were missed.
4. **Guaranteed cleanups** — bracketed vocal sounds and core fillers are always removed,
   independent of the LLM.
5. **Re-assembly** — keeps only the non-deleted words (deleted words break the block) and
   snaps each cut to a real gap in the audio, because word timestamps alone clip endings and
   let the tail of a deleted word through.
6. **Export** — a clean `wav` / `flac` / `mp3`, a **Premiere Pro project** whose sequence
   cuts the original video and audio together, and a **cut list** that shows the deleted words in
   the transcript, for human correction.

---

## Pipeline

```
audio/video ──ffmpeg──▶ 16 kHz mono ──CrisperWhisper (verbatim)──▶ words[start,end]
                                                                        │
         core fillers + bracketed sounds (always) ─────────▶ delete indices ◀┘
                                                                        │
                LLM (DeepSeek) windowed, per-word dedup ──────────▶ delete indices
                                                                        │
                LLM review of the result: restore or delete more ──▶ delete indices
                                                                        │
                        clean_keep_blocks ──▶ kept blocks (no residue)
                                                                        │
         ffmpeg streaming splice ──▶ out.wav
         FCP XML ──▶ local Premiere imports and saves ──▶ out.prproj
```

- **Transcription** runs in a separate Python interpreter that has a CUDA PyTorch.
- **Everything else** (LLM, splice, Premiere project) runs in the web-server virtual environment.

---

## Features

- **Verbatim ASR** that keeps disfluencies, so they can actually be deleted.
- **AI dedup** with a clear per-word rule (keep the last, delete the earlier), windowed
  so it stays fast and reliable on long files.
- **No residue / no clicks**: deleted segments are removed entirely, including their
  cut-padding edges.
- **LLM controls**: high reasoning effort, parallel windows, and a review pass. The
  API settings are checked before transcription starts. A window that fails is
  retried after the others finish, at lower concurrency. If it still fails, it is reported as a
  warning next to the result with its time range, never hidden. A reply that deletes more than
  60% of a window is treated as a failure, because losing speech is worse than keeping a mis-speak.
- **Fast re-runs**: the transcript of a local file is saved, so running the same file again goes
  straight to the LLM. Transcription is by far the slowest step.
- **Multiple outputs**: lossless `wav` / `flac`, high-bitrate `mp3`, and a **Premiere Pro
  project** that references the original file, with linked video and audio clips.
- **Offline model loading**: once the model is cached, it is loaded without touching the
  network.
- **Real-time server-sent progress** in the browser while processing.

---

## Requirements

- **Windows** (paths and commands below are Windows-specific, but the logic is portable).
- **Python 3.12+** for the server `venv`.
- **A CUDA-enabled Python 3.14** environment with PyTorch for transcription (see
  Installation).
- **ffmpeg / ffprobe** on `PATH` (decoding and silence detection).
- An OpenAI-compatible API key, e.g. **DeepSeek** with `deepseek-flash`. It is required: the LLM is
  what finds the mis-speaks.
- **Adobe Premiere Pro** to get a `.prproj`. Without it you get an `.xml` to import instead.

---

## Installation

### 1. Server virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs Flask, openai, and numpy.

### 2. Transcription environment (CUDA PyTorch, separate interpreter)

The verbatim transcriber (`faithful_transcribe.py`) needs a PyTorch build with CUDA.
Install it in a *separate* interpreter (e.g. your global Python 3.14):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install transformers crisperwhisper[convert] librosa soundfile soxr numpy
```

> **Note:** On Windows, the CrisperWhisper **CTranslate2 backend is not available**
> (its `ctranslate2-crisperwhisper` fork ships Linux-only wheels). The tool therefore
> uses the **transformers** backend, which works but is slower on very long files.
> The pipeline mitigates this with chunked transcription and live progress.

### 3. ffmpeg / ffprobe

Ensure `ffmpeg` and `ffprobe` are on `PATH` (e.g. installed via `winget install
Gyan.FFmpeg`).

### 4. Download the model

The first run with each model size needs internet access: it downloads the CrisperWhisper
model (e.g. `CrisperWhisper2.0_large`) into `HF_HOME`. Once that model is fully cached, it
is loaded **offline**.

---

## Configuration

- **Model**: the default is `large` (most accurate); `small` is available for speed.
- **API**: enter the OpenAI-compatible API URL (e.g. `https://api.deepseek.com`), model
  (e.g. `deepseek-flash`), and your API key in the web form.
- **Dedup** favors quality over speed and cost. It requests high reasoning effort
  (`REASONING_EFFORT` in `dedup.py`), and sends the transcript in **windows** of 300 words with
  150 words of read-only context on both sides. It sets no output token limit, so the answer is
  not truncated by a small budget. If a provider rejects a parameter, the request is retried
  without it, giving up JSON mode before reasoning. If no reasoning setting is accepted, the result
  carries a warning.
- **Review** runs once the first pass has finished, with its window boundaries shifted by half a
  window, so words near a first-pass boundary sit in the middle of a review window. Windows in each
  pass run in parallel and can finish in any order: each one reads the same input and changes only
  its own words, so the result does not depend on the order.

---

## Usage

### Start the server

Double-click `start.bat` (creates the venv-backed server and opens the browser), or run:

```powershell
.\.venv\Scripts\python.exe server.py
```

Then open **http://127.0.0.1:7861**.

### In the web UI

1. Enter the **local file path** (e.g. `F:\videos\talk.mp4`) or choose a file to upload.
   Use the path for large files: the source is read in place, with no upload or copy,
   and it is never modified or deleted.
2. Fill in the **API URL**, **model name**, and **API key**. The page remembers your settings for
   next time, except the key and the file path.
3. Choose the **Whisper model** (`large` default) and output settings.
4. Close Premiere if it is open, then click **Start Processing**. Watch the live progress bar.
5. Download the clean audio, and open the **`.prproj`** from the `output` folder.

### The Premiere project

- At the end of a run, Premiere opens by itself for a short time, imports the edit, saves the
  `.prproj`, and closes. Avoid using Premiere during that moment.
- Open the `.prproj`: the sequence is already in the timeline, with each kept segment as a
  linked video and audio clip, for easy manual correction.
- It references the **original source file** when you used the local path. An uploaded file is
  deleted after processing, so relink it by name when prompted.
- If Premiere was open or is not installed, you get an **`.xml`** and a warning instead. Use
  `File > Import` in Premiere: the sequence is the same.

---

## Outputs

- **`wav` / `flac`** — lossless, preserving the source sample rate and channels. FLAC is the default.
- **`mp3`** — high-bitrate (default 320k) compressed.
- **`m4a`** — AAC, when "Match source" is selected and the source audio is AAC.
- **`.prproj`** — Premiere Pro project (all kept segments, video and audio, referencing the source).
- **`.xml`** — the same sequence for `File > Import`, when a `.prproj` could not be created.
- **`<name>_cuts.txt`** — the transcript by sentence, with deleted words marked `[-like this-]`.
  Words only the review deleted are marked `[--like this--]`, and words the first pass deleted but
  the review kept are marked `[+like this+]`. Each line starts with its time in the edited sequence and in the source, so you can read
  through the decisions and jump to the ones you want to fix in Premiere.

`output\` contains only these finished results. Intermediate files, temp files and the
debug log live in `.work\`; each job's files are removed when it ends, and anything left
by a forced shutdown is removed the next time the server starts.

Transcripts of local files stay in `.work\transcripts\` (about 1 MB per hour of audio). A saved
transcript is used only if the file (path, size, modified time), the Whisper model, the language
and `faithful_transcribe.py` are all unchanged. Delete the folder at any time to transcribe again.
Uploaded files are not saved this way.

---

## Project layout

```
Root Folder
  server.py              # Flask web server + SSE progress + upload/static
  dedup.py               # dedup engine: transcription orchestration, LLM dedup,
                         #   keep-block building, splice
  premiere.py            # Premiere project: writes FCP XML, has Premiere save it as .prproj
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

- **"CrisperWhisper failed (exit 1): ..."** — the message after the colon is the cause. If
  the model cannot be loaded on a first run, check the internet connection: a model that is
  not fully cached has to be downloaded into `HF_HOME`.
- **"LLM API check failed"** — the URL, model name, or key was rejected. Nothing has been
  transcribed yet, so fix the form and run again.
- **Warning: "LLM dedup failed for N of M windows"** — the audio was still produced, but
  mis-speaks in the listed time ranges remain. If every window fails, the job stops with an error
  instead. Check `.work\debug.log` for the `window ... failed` lines. Running the file again
  reuses the transcript, so it only costs the LLM step.
- **Warning: "The LLM review failed for N of M windows"** — the first-pass cuts in the listed time
  ranges were used without the second check. Everything else is complete.
- **Warning: "No .prproj was created"** — the message gives the reason. Import the `.xml` that
  was saved instead, or fix the cause and run again.
- **Warning: "Premiere cannot read this file format"** — Premiere does not open every container
  (MKV is the common case, and it is OBS's default). Convert it first, then run on the MP4:
  `ffmpeg -i input.mkv -c copy output.mp4`.
- **Clicks / residue at cut points** — each cut is snapped to a quiet gap near the word's
  timestamp. If words run together there is no gap to find, so that cut stays on the timestamp
  and can still sound abrupt. The search constants are at the top of the cut-refinement code
  in `dedup.py`.
- **Slow on long files** — the transformers backend is slower than CTranslate2 on
  Windows; the tool chunk-transcribes and streams progress so it is never "stuck". Jobs run
  one at a time; a second job waits in the queue. Writing AAC or MP3 also takes minutes on a long
  file (AAC at 320k runs at about 16x real time).
- **Debug log** — look in `.work\debug.log` for transcription, LLM replies, and the
  final keep/delete ranges.

---

## License & notes

- The transcription model (CrisperWhisper) is licensed for **non-commercial research**.
  Use it for your own personal projects accordingly.
- This project is provided as-is for local, personal use.
