# Future Extension Plan: Generative Speech Editing with Tencent AuK

> **Status:** Proposed / not implemented — roadmap document.
> **Idea:** add an *optional generative editing mode* that removes mis-speaks and
> fillers by **regenerating** the speech with a speech-editing foundation model
> (Tencent **AuK**), as an alternative to the current non-destructive cut-and-splice.

---

## 1. Motivation

The current pipeline removes mis-speaks by **cutting time ranges out of the original
recording and splicing the kept parts back together**:

```
verbatim ASR → LLM dedup (word indices) → clean_keep_blocks → splice (cut) → WAV + AAF
```

This is lossless with respect to the speaker's voice, but it has an inherent weakness:
**every cut is a hard edit**. We have already fought several artifacts caused by this —
padding residue, overlapping keep blocks (duplicated audio), and pad clamp issues at
cut boundaries. In other words, *cut-and-splice can never be perfectly seamless*,
because the removed material leaves an audible edit point.

A **generative speech editor** removes the same material differently: it is told
*"remove this phrase"* (or *"remove the breaths and coughs"*) and then **regenerates
the audio seamlessly**. There is no edit point to hide, because the model produces a
new, continuous waveform.

**Goal:** keep the current cut-based mode as the default (lossless, fast, deterministic,
Premiere-AAF friendly), and add generative editing as an **opt-in mode** that trades a
little voice fidelity for a **seamless** result.

---

## 2. What AuK is (relevant summary)

**AuK** — *An Open-Source Foundational Model for Speech Generation and Editing* —
Tencent Hunyuan, open-sourced 2026-09-09, **MIT-licensed**.

- **1.5B** foundation model, trained on millions of hours of audio.
- **Unified natural-language instruction interface** for all tasks.
- Two variants:
  - **AuK** (base): higher quality, configurable NFE / CFG.
  - **AuK-Flash**: distilled, **4-step** inference (NFE=4, CFG=0).
- **Architecture:** Qwen2.5-Omni-3B (MLLM encoder) + Diffusion Transformer + VAE.

### Tasks that matter for this project

| AuK task | Relevance here |
|---|---|
| **Speech Content Editing** — replace / insert / **remove** text | Remove a mis-spoken phrase and regenerate seamlessly (replaces the cut) |
| **Nonverbal Editing** — remove/add breaths, laughs, coughs | Removes `[throatclearing]`, `[UM]`, `[cough]`, breaths — the tokens our verbatim ASR preserves |
| **Speech Enhancement** — denoise / dereverberate / restore | Optional cleanup pass before/after editing |
| Pitch / Speed / Volume, Emotion, Timbre, De-accent | Not needed now, but available through the same interface |
| Zero-shot TTS / Instruct TTS | Not needed for dedup; possible future "patch the audio by re-speaking a line" feature |

- Repo: <https://github.com/Tencent-Hunyuan/AuK>
- Weights: `tencent/AuK`, `tencent/AuK-Flash`, encoder `Qwen/Qwen2.5-Omni-3B`
- Technical report: arXiv 2609.08936
- **License: MIT** (permissive — unlike the non-commercial CrisperWhisper weights)

---

## 3. Design: two dedup modes

Add a **`dedup_method`** selector to the web UI and to `dedup_audio(...)`:

| Mode | How it works | Pros | Cons |
|---|---|---|---|
| **`cut`** (default, current) | Delete word spans, splice original audio, export AAF | Lossless voice, fast, deterministic, editable in Premiere | Hard edit points; needs padding/clamp care |
| **`generative`** (new) | Identify mis-speak spans, instruct AuK to remove/regenerate | **Seamless** — no edit points | Generated voice (slight timbre drift), slower, GPU-heavy; no meaningful AAF |
| **`hybrid`** (optional) | Cut for large deletions, AuK only for small in-word touch-ups | Best of both | Most complex |

### 3.1 Where it plugs in

The front half of the pipeline is **unchanged** — it is already exactly what a
generative editor needs:

```
faithful_transcribe.py (verbatim ASR, word timestamps)
        │
        ▼
llm_dedup_words()  ──►  deleted word indices  →  deleted spans (start, end, text)
        │
        ├── cut        → clean_keep_blocks → _splice_keep → WAV / FLAC / MP3 + AAF
        │
        └── generative → build AuK instruction(s) → AuK edit → rendered WAV
```

### 3.2 Mapping our deletion plan → AuK instructions

AuK edits are driven by **natural-language instructions that reference the spoken
text**, not by time ranges. So the adapter must turn our word-index deletion plan into
instruction text:

1. **Group** consecutive deleted indices into contiguous *mis-speak spans*.
2. For each span, recover its **transcript text** (join the deleted words).
3. Emit an instruction, e.g.
   - Content: `Remove "inspired by New York City, inspired by New York City," from the speech.`
   - Nonverbal: `Remove the breath, cough, and filler sounds (um, uh) from this audio.`
   - Filler-only span → prefer the **Nonverbal Editing** task.
4. Compute the **expected output duration** (`gen_seconds`) = original duration − sum of
   removed spans, since AuK takes a target duration hint.

> Note: many small scattered deletions are easier to express as **one non-verbal /
> cleanup instruction** over the whole window than as dozens of per-word removals.
> The adapter should **merge nearby spans** and prefer one instruction per window.

### 3.3 Long audio → windowing and stitching (the main engineering work)

AuK's ComfyUI integration documents a **≈30 s limit** on the source+target sequence, and
the model is trained on short segments. A 47-minute narration therefore needs:

1. **Windowed processing** — split into ~20–30 s windows at natural pauses (reuse the
   existing silence/pause detection).
2. **Context margin** — include a small overlap/context so the model does not cut words
   at the window edge.
3. **Stitching** — concatenate the regenerated windows with a short crossfade.
   *(This is itself a hard-cut problem — but at a much coarser granularity, ~2–3
   boundaries per minute, rather than per removed word.)*
4. **Long-term alternative:** if AuK gains a streaming/longform mode, drop the manual
   windowing.

### 3.4 Isolation from the current environments

There are already **two** Python environments in this project:

- the server **venv** (Python 3.12) — Flask, openai, pydub, pyaaf2;
- the **transcription** interpreter (Python 3.14 + CUDA torch) — CrisperWhisper.

AuK requires **Python 3.10** (uv/conda) with its own torch stack, and possibly a
different CUDA build. It must therefore be a **third, isolated environment**, invoked
the same way `faithful_transcribe.py` is: a **subprocess** (or a small local HTTP
service) with a clean environment. Do **not** try to install AuK into the existing
venv or the 3.14 interpreter.

```
auk_edit.py            # new adapter script, runs in the AuK env (Python 3.10)
  in : <audio.wav> <plan.json>   (deleted spans + text + target duration)
  out: <edited.wav>
```

`dedup.py` then shells out to it exactly like it does for transcription, streaming
progress back to the UI.

---

## 4. Dependencies / prerequisites

- **AuK env:** Python 3.10; `pip install -e ".[gradio]"` (or core only) from the AuK repo.
- **Weights:** `tencent/AuK` (quality) and/or `tencent/AuK-Flash` (speed), plus the
  `Qwen/Qwen2.5-Omni-3B` MLLM encoder and the VAE.
- **GPU:** RTX 5070 Ti (16 GB) — AuK 1.5B + Qwen2.5-Omni-3B + VAE should fit; **AuK-Flash**
  (4-step) is the likely default for interactive use.
- **Disk:** model weights are on the order of several GB — the `F:` drive has been tight;
  account for this.
- **License:** MIT (weights + code) — commercially usable, unlike the ASR model.

---

## 5. Milestones

### P0 — Feasibility spike (no integration)
- [ ] Create the isolated AuK Python 3.10 env; download `AuK-Flash` + `Qwen2.5-Omni-3B`.
- [ ] Run the CLI on a **short clip with a known mis-speak**:
      `auk-infer --audio clip.wav --instruction "Remove \"<phrase>\" from the speech." --output out.wav --gen_seconds N`
- [ ] Also try **Nonverbal Editing**: *"Remove breaths, coughs and filler sounds."*
- [ ] Measure: **quality** (timbre drift? artifacts?), **latency** per 30 s window, **VRAM**.
- [ ] **Decision gate:** is the seamless result actually better than the cut result? If
      not, stop here and keep this document as a "considered and rejected" note.

### P1 — Adapter (`auk_edit.py`)
- [ ] Define the plan JSON: `{windows:[{start,end}], removals:[{start,end,text,kind}], gen_seconds}`.
- [ ] Build `dedup.py` → instruction-text mapping (group spans, merge nearby, recover text,
      classify content vs nonverbal).
- [ ] Chunk long audio into windows at pauses; stitch with crossfade.
- [ ] Emit a rendered WAV; write a debug log like the existing `output/debug.log`.

### P2 — Pipeline + UI wiring
- [ ] Add `dedup_method` ("cut" | "generative") to the Flask form and to `dedup_audio`.
- [ ] Route the pipeline: `generative` → `auk_edit.py` instead of `_splice_keep`.
- [ ] Progress reporting (reuse the existing SSE + subprocess-reader pattern).
- [ ] Keep AAF export for `cut`; for `generative`, return the rendered WAV (document why
      AAF is not meaningful for regenerated audio).

### P3 — Quality, speed, evaluation
- [ ] Objective + subjective A/B: cut vs generative vs original, on a fixed test set.
- [ ] Try `AuK` vs `AuK-Flash`; tune NFE/CFG vs speed.
- [ ] Optional **Prompt Enhancer** integration (turns a free-form request into a proper
      AuK instruction + duration automatically).
- [ ] Optional **fine-tuning** on the user's own narration style (AuK ships a JSONL
      fine-tuning pipeline).
- [ ] Consider `hybrid`: cut for long deletions, AuK for small touch-ups.

---

## 6. Risks & open questions

| Risk | Notes / mitigation |
|---|---|
| **Voice timbre drift** | Generative edits may subtly change the speaker's voice. Mitigate with reference-audio conditioning; A/B test. Accept only if seamless-ness wins. |
| **Long-audio seams** | Windowing reintroduces cut points at window boundaries. Mitigate with pause-aligned windows + crossfade; measure seam audibility. |
| **Hallucinated / altered content** | The model could regenerate words that were not meant to change. Verify by re-transcribing the output and diffing against the expected transcript. |
| **Duration control** | `gen_seconds` must be accurate or output length drifts. Compute from transcript timestamps. |
| **Latency / VRAM** | Diffusion inference is far slower than cutting. AuK-Flash (4-step) is the practical default; batch windows where possible. |
| **Env complexity** | A third Python env is real maintenance cost. Consider a single local HTTP service wrapping AuK, so the server talks to it over `localhost`. |
| **Determinism** | Diffusion is stochastic; cut is deterministic. Generative output may vary run-to-run — pin seeds if reproducibility is needed. |
| **Not a Premiere workflow** | Generative output is a rendered file, so the "import AAF and correct by hand in Premiere" workflow does not apply in this mode. |

---

## 7. Acceptance criteria (for P2)

1. A user can pick **generative** mode in the UI and get a rendered WAV with the same
   mis-speaks removed as the cut mode would remove.
2. On a 30 s test clip with a mid-sentence mis-speak, the generative output has **no
   audible edit point** where the cut mode has one.
3. The output duration is within ±0.5 s of (original − removed).
4. Re-transcribing the output shows the mis-speak phrase is gone and **no new content**
   was introduced.
5. The existing **cut** mode is unchanged and remains the default.

---

## 8. References

- AuK repository: <https://github.com/Tencent-Hunyuan/AuK>
- Demo: <https://huggingface.co/spaces/tencent/AuK> · <https://modelscope.cn/studios/Tencent-Hunyuan/AuK>
- Technical report: arXiv 2609.08936
- Cookbook (instruction templates per task): `docs/COOKBOOK.md` in the AuK repo
