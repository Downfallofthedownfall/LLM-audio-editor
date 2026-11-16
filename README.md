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
   earlier repeats"** rule, and returns the indices to delete.
4. **Guaranteed cleanups** — bracketed vocal sounds and core fillers are always removed,
   independent of the LLM.
5. **Line check: off-script remarks and earlier takes** — the LLM then reads what is left, one
   sentence per line, as an editor would, and looks for whole lines that do not belong in the
   video: the speaker talking to themselves (`my god it's laggy, why?`), notes about the recording
   (`I'll record the voice after`), and an earlier take of something that is said again later in
   other words. Every line is read twice, each time with different surrounding text. A line is cut
   only when both readings agree. The rest stay in the edit and are marked for you to check.
6. **Re-assembly** — keeps only the non-deleted words (deleted words break the block) and
   snaps each cut to a real gap in the audio, because word timestamps alone clip endings and
   let the tail of a deleted word through.
7. **ASR review: listening to the edit** — renders the cut audio and transcribes *it* again,
   word by word, then compares what is heard with what was planned. This finds what the
   transcript cannot show: the start or end of a deleted word that leaked through a cut, a kept
   word that a cut chopped (`situation` heard as `situa-`), and mis-speaks the first transcription
   missed. Leftovers are trimmed, chopped words get their cut moved to the next real gap, and the
   LLM reads the text that was actually heard to restore wrongly deleted words and delete what
   should not be there. By default the edit is listened to twice, so the fixes are checked too.
8. **Export** — a clean `wav` / `flac` / `mp3`, a **Premiere Pro project** whose sequence
   cuts the original video and audio together and marks the passages worth checking, and a
   **cut list** that shows the deleted words in the transcript, for human correction.

---

## Pipeline

```
audio/video ──ffmpeg──▶ 16 kHz mono ──CrisperWhisper (verbatim)──▶ words[start,end]
                                                                        │
         core fillers + bracketed sounds (always) ─────────▶ delete indices ◀┘
                                                                        │
                LLM (DeepSeek) windowed, per-word dedup ──────────▶ delete indices
                                                                        │
     LLM line check, two readings: off-script lines, earlier takes ──▶ cut lines + marked lines
                                                                        │
                        clean_keep_blocks ──▶ kept blocks ──▶ draft edit
                                                                        │
      ┌──── ASR review, repeated for each pass ─────────────────────────┤
      │  ffmpeg splice of the draft ──CrisperWhisper (verbatim)──▶ heard words
      │  align heard words with the planned words
      │  rules: trim leftovers and vocal sounds, widen cuts that chop a word
      │  LLM reads what was heard: restore or delete more
      └──────────────────────────────────────────────▶ revised kept blocks
                                                                        │
         ffmpeg streaming splice ──▶ out.wav
         FCP XML with markers ──▶ local Premiere imports and saves ──▶ out.prproj
         word-by-word edit record ──▶ .work\edits\out.json ──▶ evaluate.py
```

- **Transcription** (of the source, and of the edit in each review pass) runs in a separate
  Python interpreter that has a CUDA PyTorch.
- **Everything else** (LLM, splice, Premiere project) runs in the web-server virtual environment.

---

## Features

- **Verbatim ASR** that keeps disfluencies, so they can actually be deleted.
- **AI dedup** with a clear per-word rule (keep the last, delete the earlier), windowed
  so it stays fast and reliable on long files.
- **Off-script remarks and earlier takes**: the line check removes what a per-word rule cannot
  see. It exists because of a measurement. On an 85-minute recording that was then corrected by
  hand in Premiere, the per-word pass had removed 1,624 of the 2,591 words that had to go (63%),
  and none of its deletions was put back. The 967 words it left were in 79 places, almost all of
  them off-script remarks or earlier takes of a whole sentence. How many of these the line check
  finds depends on the LLM and has not been measured yet: `evaluate.py` measures it once you have
  corrected a project (see below).
- **A short list to check instead of the whole edit**: every line the line check cut, and every
  line it was not sure about, has a marker in the Premiere sequence and an entry at the top of the
  cut list.
- **No residue / no clicks**: deleted segments are removed entirely, including their
  cut-padding edges.
- **The result is listened to, not only read**: the edit is transcribed again and compared
  with the plan, so leftovers and chopped words at the cuts are found and fixed. Measured on an
  85-minute recording whose edit had 353 cuts: the first listen trimmed 41 leftover sounds and
  moved 19 cuts that chopped a word. The second listen heard 14 of the 17 chopped words in full,
  undid 3 trims that had cut into a kept word, and changed 4 other places.
- **LLM controls**: high reasoning effort, parallel windows, and a review of the heard text. The
  API settings are checked before transcription starts. A window that fails is
  retried after the others finish, at lower concurrency. If it still fails, it is reported as a
  warning next to the result with its time range, never hidden. A reply that deletes more than
  60% of a window is treated as a failure, because losing speech is worse than keeping a mis-speak.
- **Fast, repeatable re-runs**: the transcript of a local file is saved, so running the same file
  again goes straight to the LLM. Transcription is by far the slowest step. The LLM's replies are
  saved too, so the same request gets the same answer: after a change to a later step, any
  difference in the result comes from that change.
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
  not truncated by a small budget. The API check before transcription settles the parameters: if
  the provider rejects one, the check is retried without it, giving up JSON mode before reasoning.
  If no reasoning setting is accepted, the result carries a warning. The settings found by the
  check are then used for every window. A later request can only give up JSON mode; it never
  drops reasoning to get an answer, and a window that still fails is retried and then reported.
- **Line check** reads the words the dedup kept, split into lines at sentence ends, at existing
  cuts and at pauses of 0.7 s or more. The LLM names each line it would cut as off-script or as an
  earlier take, and for an earlier take it must name the later line that replaces it. It can also
  list a line as "unsure".
  - Every line is read `TAKE_READINGS` times (`dedup.py`, default 2) in windows of about 600 words
    with 300 words of context, and the windows of the second reading start half a window later.
    A recording short enough to fit in one window is asked the same question twice instead,
    which is a weaker check.
  - A line is cut only when every reading would cut it. A line that only one reading would cut, or
    that a reading lists as unsure, stays in the edit and is marked. A missing sentence is hard to
    notice in the finished video and a leftover one is easy to hear, so in doubt the line stays.
  - An earlier take is cut only if the line that replaces it stays, so the takes of one passage are
    never all cut.
  - The LLM review cannot restore a line cut here.
  - The web form has three settings: cut and mark (default), only mark, or off. With "off" the
    edit is the same as before the line check existed.
- **Saved LLM replies**: a reply that passed the checks is saved in `.work\llm_cache\`, keyed by
  the API URL, the model, the request settings and the full request text. Running the same file
  again with the same settings reuses the replies, so the result is the same. Delete the folder to
  ask the LLM again. A reply that was refused (bad format, deleted too much) is never saved.
- **ASR review** listens to the edit `REVIEW_PASSES` times (`dedup.py`, default 2; the web form
  offers 1 to 3). Each pass transcribes the whole edit again, on top of the first transcription:
  11 to 16 minutes for a 27-minute edit on an RTX 5070 Ti. What a pass changes can only be heard in the next
  one: the second pass is what checks the fixes of the first. A pass that changes nothing ends the
  review early, and the summary line in the result says how many changes of the last pass were not
  listened to again.
  - A cut that was listened to and sounded as planned is frozen: later passes reuse it exactly.
  - A trim is checked by the next pass. If the kept word next to it can no longer be heard, the
    trimmed sound was part of that word (the recognizer sometimes hears `station` as `station dot`),
    and the trim is undone.
  - Each cut is trimmed at most once. After a trim the recognizer can mishear the new join, and
    trimming again on that evidence would cut further into the kept word with every pass.
  - Two transcriptions of the same audio differ in about 0.5% of the words (`want to` / `wanna`).
    A different spelling alone never changes the edit.
  - The review uses the **local CrisperWhisper**, not a cloud ASR. It needs verbatim output
    (a normalizing recognizer drops the stray `uh` or `s-` that the review looks for) and word
    timestamps (to map a leftover back to its cut). DeepSeek's speech-to-text API returns plain
    text without timestamps and accepts at most 120 seconds per request.
- **LLM review** reads the heard text, with its window boundaries shifted by half a window, so
  words near a first-pass boundary sit in the middle of a review window. Windows in each
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
3. Choose the **Whisper model** (`large` default), what the **line check** does with off-script
   lines and earlier takes, how many **ASR review** passes to run, and the output settings.
4. Close Premiere if it is open, then click **Start Processing**. Watch the live progress bar.
5. Download the clean audio, and open the **`.prproj`** from the `output` folder.
6. Go through the markers in the sequence (see below), then watch the edit once.

### The Premiere project

- At the end of a run, Premiere opens by itself for a short time, imports the edit, saves the
  `.prproj`, and closes. Avoid using Premiere during that moment.
- Open the `.prproj`: the sequence is already in the timeline, with each kept segment as a
  linked video and audio clip, for easy manual correction.
- The sequence has a marker at every place the line check touched. Jump between them with
  `Shift+M` and `Ctrl+Shift+M`:
  - **`Cut: off-script`** / **`Cut: earlier take`** sits on the cut where whole lines were removed.
    The marker's comment holds the removed text. To bring it back, drag the edge of the clip next
    to the marker outwards.
  - **`Check: …?`** spans a passage that was kept although the line check suspects it. The passage
    is a clip of its own, with the Mango label colour, so you can select it and ripple-delete it.
    Its edges sit in the middle of the pauses around it. If you renamed Premiere's label colours,
    the clip keeps the default colour and only the marker shows it.
- The same places are listed under "Check first" at the top of the cut list.
- The markers and the label colour travel in the XML. This was checked with Premiere Pro 2025:
  both come through the import. If a Premiere version drops the markers on import, the tool
  creates them in Premiere before it saves the project.
- It references the **original source file** when you used the local path. An uploaded file is
  deleted after processing, so relink it by name when prompted.
- If Premiere was open or is not installed, you get an **`.xml`** and a warning instead. Use
  `File > Import` in Premiere: the sequence is the same.
- The sequence counts in video frames, so each cut is rounded to the nearest frame and can sit
  up to half a frame (21 ms at 24 fps, 17 ms at 30 fps) away from the cut in the audio file. The
  audio file is the version the ASR review listened to. A cut in a pause has 30 to 40 ms of room
  around the word, which covers this. A cut between two words spoken without a pause has none, so
  listen to those in the sequence.
- If the audio track of the file starts later than the file itself, the sequence times are
  shifted by that offset, so the cuts land on the same sound as in the audio file.

---

## Outputs

- **`wav` / `flac`** — lossless, preserving the source sample rate and channels. FLAC is the default.
- **`mp3`** — high-bitrate (default 320k) compressed.
- **`m4a`** — AAC, when "Match source" is selected and the source audio is AAC.
- **`.prproj`** — Premiere Pro project (all kept segments, video and audio, referencing the source).
- **`.xml`** — the same sequence for `File > Import`, when a `.prproj` could not be created.
- **`<name>_cuts.txt`** — the cut list. It starts with a key to the marks and a summary of what
  the line check and the ASR review did. Then comes **"Check first"**: the passages the line check
  cut or was unsure about, each with its time in the edit and in the source. Then the full
  transcript by sentence:
  - `[-like this-]` — deleted by the first pass (mis-speaks, restarts, repeats, filler sounds).
  - `[=like this=]` — whole lines cut by the line check.
  - `[?like this?]` — kept, but the line check suspects it is off-script or an earlier take.
  - `[--like this--]` — deleted by the review: mis-speaks the first pass missed, and leftover
    sounds it heard at the cuts.
  - `[+like this+]` — deleted by the first pass, kept by the review.

  The text also contains words that were only heard in the edit, which the first transcription
  missed. Each line starts with its time in the edited sequence and in the source, so you can read
  through the decisions and jump to the ones you want to fix in Premiere.

`output\` contains only these finished results. Intermediate files, temp files and the
debug log live in `.work\`; each job's files are removed when it ends, and anything left
by a forced shutdown is removed the next time the server starts.

Each run also leaves an **edit record** in `.work\edits\<name>.json`: every word with its
timestamps and what was decided about it, and the kept ranges. `evaluate.py` reads it.

Transcripts of local files stay in `.work\transcripts\` (about 1 MB per hour of audio). A saved
transcript is used only if the file (path, size, modified time), the Whisper model, the language
and `faithful_transcribe.py` are all unchanged. Delete the folder at any time to transcribe again.
Uploaded files are not saved this way.

---

## Measuring the result against your own correction

After you have corrected an edit in Premiere and saved the project, that project is the answer
key: what is still in the sequence should have been kept, and the rest should have gone.
`evaluate.py` compares it with what the tool decided, word by word:

```powershell
.\.venv\Scripts\python.exe evaluate.py .work\edits\talk_deduped.json output\talk_deduped.prproj
```

It reports:

- **Missed**: words the tool kept and you removed, with the text of each place, and how many of
  them the line check had at least marked.
- **Wrong**: words the tool removed and you put back, by the step that removed them.
- **Cut points**: how many clip edges you moved by up to half a second.

The answer key is stored in source time, so it also scores any later run on the same file. Change
a prompt or a rule, run the file again (the transcript is reused), and compare the new edit record
with the same project. Add `--save-truth .work\truth\talk.json` once to keep the answer key
after the project changes, and pass that file instead of the project from then on.

Differences of a single word are counted apart and left out of the scores. Most of them come
from timestamps, where a word's time range includes the pause before it.

### Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The tests need no GPU, LLM or Premiere. They cover the line check with a stand-in LLM, the saved
replies, the cut list, the Premiere XML, the evaluation, and one property of the ASR review: when
the edit is heard exactly as planned, no cut moves.

---

## Project layout

```
Root Folder
  server.py              # Flask web server + SSE progress + upload/static
  dedup.py               # dedup engine: transcription orchestration, LLM dedup, line check,
                         #   keep-block building, ASR review of the edit, splice
  premiere.py            # Premiere project: writes FCP XML with markers, has Premiere save it as .prproj
  faithful_transcribe.py # verbatim transcription subprocess (runs in global CUDA env)
  evaluate.py            # compares an edit with a Premiere project corrected by hand
  tests/                 # tests that need no GPU, LLM or Premiere
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
- **"CrisperWhisper crashed (0xC0000005, access violation)"** — the transcriber was killed by
  Windows instead of reporting an error; the tool has already started it a second time. The usual
  cause is memory: the large model needs about 17 GB of free commit (RAM + page file), mostly
  while the first chunk runs, and PyTorch crashes instead of raising an error when Windows cannot
  provide it. Close memory-heavy apps (Premiere, browsers, games), or give the page file a larger
  fixed size on a drive with free space, then run again. `.work\debug.log` records the free
  memory at the start of each attempt and the Python stack at the crash.
- **"LLM API check failed"** — the URL, model name, or key was rejected. Nothing has been
  transcribed yet, so fix the form and run again.
- **Warning: "LLM dedup failed for N of M windows"** — the audio was still produced, but
  mis-speaks in the listed time ranges remain. If every window fails, the job stops with an error
  instead. Check `.work\debug.log` for the `window ... failed` lines. Running the file again
  reuses the transcript, so it only costs the LLM step.
- **Warning: "The line check failed for N of M windows"** — in the listed time ranges a line was
  read once instead of twice, or not at all. Lines read once are only marked, never cut, so
  off-script lines and earlier takes may remain there. Running the file again asks only for the
  windows that failed: the other replies are saved.
- **The line check cut something that belongs in the video** — find its `Cut:` marker in the
  sequence and drag the clip edge back out. If it happens often with your material, choose "Only
  mark them" in the form, and tell the rules in `TAKE_RULES` (`dedup.py`) what to keep.
- **Warning: "the N markers for the passages to check could not be added"** — the project itself
  is complete. Use the "Check first" list at the top of the cut list instead.
- **Warning: "The LLM review failed for N of M windows in pass P"** — the cuts in the listed time
  ranges were used without the LLM's check of that pass. Everything else is complete.
- **Warning: "The edit could not be listened to"** — the transcriber failed on the cut audio (the
  message gives the reason, usually memory, see the crash entry above). The LLM still reviewed the
  transcript, but leftovers and chopped words at the cuts were not checked. Run the file again.
- **Warning: "Review pass N could not listen to the edit"** — the same failure in a later pass. The
  result contains the changes of the pass before it, which were not listened to.
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
  Windows; the tool chunk-transcribes and streams progress so it is never "stuck". Each ASR
  review pass transcribes the edit again, and the edit is dense speech, so a pass takes roughly
  half as long as the edit lasts. Choose 1 review pass in the form to save one of them. Jobs run
  one at a time; a second job waits in the queue. Writing AAC or MP3 also takes minutes on a long
  file (AAC at 320k runs at about 16x real time).
- **Debug log** — look in `.work\debug.log` for transcription, LLM replies, and the
  final keep/delete ranges.

---

## License & notes

- The transcription model (CrisperWhisper) is licensed for **non-commercial research**.
  Use it for your own personal projects accordingly.
- This project is provided as-is for local, personal use.
