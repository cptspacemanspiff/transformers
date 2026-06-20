<!--
Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Sortformer speaker diarization

[Sortformer](https://huggingface.co/docs/transformers/model_doc/sortformer) is an end-to-end speaker-diarization model:
for every output frame it emits an independent probability for each of up to `num_speakers` speakers being active
(multi-label / sigmoid). It is a *streaming* model — it keeps a bounded **Arrival-Order Speaker Cache (AOSC)** so memory
and compute stay constant no matter how long the audio is — and it labels speakers by **arrival order** (the first
speaker to talk is speaker 0, the next is speaker 1, …), kept consistent across the whole stream by the speaker cache.

## Scripts

| Script | What it does |
| --- | --- |
| `run_sortformer_diarization.py` | Diarize a wav file + a plot: offline, `--streaming` (AOSC), or `--streaming --causal` (cache-aware, see below). |
| `run_sortformer_microphone.py` | Live mic diarization via `streaming_step` (re-encodes the whole window each step). |
| `benchmark_sortformer.py` | Throughput / latency benchmark. |

```bash
pip install transformers torch soundfile sounddevice

python run_sortformer_diarization.py --audio meeting.wav
python run_sortformer_diarization.py --audio meeting.wav --streaming --causal --right-context 7
```

All streaming paths expect features extracted **without** waveform peak-normalization (the offline forward peak-
normalizes; streaming does not), which the scripts handle for you.

---

## How the incremental streaming encoder works

This is the design behind `streaming_step_incremental` / `diarize_streaming_incremental` (in
`models/sortformer/modeling_sortformer.py`). It is an inference-only, drop-in alternative to `streaming_step` that
makes the per-step cost flat in the buffer sizes.

### Background: the buffers and the expensive op

Sortformer streaming keeps two buffers of **subsampled conformer-input** embeddings:

- **FIFO** — the most recent `fifo_len` frames (recent acoustic context).
- **Speaker cache** — up to `spkcache_len` salient frames selected from *all* history (the long-term speaker-identity
  memory that keeps speaker labels stable across silences).

Each step, the model must produce diarization predictions for the new chunk by running the **17-layer FastConformer**
over `[speaker_cache + fifo + chunk]`, then a Transformer head on the result. The plain `streaming_step` re-runs the
**whole** conformer over that window every step, so the cost grows with `spkcache_len + fifo_len` (~190 frames). That
full re-encode is the expensive part; everything below is about not paying it every step.

### Key fact: causal K/V are final once computed

The streaming conformer is **causal** (each frame attends to its past + a small intra-chunk look-ahead, and the
depthwise convs are left-padded). So a frame's per-layer projected key/value vectors are **final the moment it is
encoded** — they never change again. That is exactly the property a KV cache needs: we can encode a chunk once, keep
its per-layer K/V, and have later chunks attend to them without re-encoding. (This is the same trick as a decoder KV
cache, applied to the conformer encoder.)

So the state (`SortformerStreamingIncrementalState`) holds, for both the speaker cache and the FIFO: the input
embeddings, the per-layer **frozen K/V**, the conformer **outputs** (for the head), and the predictions; plus a rolling
causal-conv left-context.

### Positions are renumbered, not absolute

Sortformer uses Transformer-XL-style **relative** position attention. NeMo lays `[speaker_cache, fifo, chunk]` out as
one **contiguous** position run — the cache is treated as sitting immediately before the FIFO, *not* at the frames'
true (possibly minutes-old) timestamps. That keeps the cache↔chunk distances small and bounded (≤ `spkcache_len +
fifo_len`), which is the regime the model was trained on. The incremental path does the same: every step it lays the
cache out contiguously before the chunk and recomputes only the relative-position bias (which never depended on the
cache *content*), so the frozen K/V stay valid. Feeding the frames' true distances instead would be out-of-distribution
(those distances grow without bound) — renumbering is required, not optional.

### The loop

For each incoming chunk, the step is one of two kinds:

**Incremental step (cheap, the common case).** Run the chunk through the 17 layers **once**, attending to the frozen
`[speaker_cache + fifo]` K/V; append its K/V / outputs to the FIFO; slide the FIFO so it stays full at `fifo_len`
(the oldest chunk's frames drop out). Cost ≈ one chunk through the encoder — flat, independent of how long the stream
has run.

**Compression step (periodic, every `compression_period` frames).** When a chunk would push the frame count past the
period, instead of encoding it incrementally:

1. **Select** the `spkcache_len` most salient input-embedding frames from `[speaker_cache + only the FIFO frames that
   arrived since the last compression]` into the new speaker cache. Each FIFO frame is considered for promotion
   exactly once; older frames already had their turn and remain recent context only.
2. **Insert** the new chunk into the FIFO (folding any frame that slides out into the running silence profile).
3. **Re-encode** `[new speaker_cache + fifo]` (now including the chunk) **from scratch** under renumbered contiguous
   positions, and re-seed every cache (per-layer K/V, outputs, conv left-context) from that pass.

The re-encode re-contextualizes the cache frames against the current window (closer to the offline computation) and
resets positions into the trained band. It is the only expensive op, and it runs ~once per period rather than every
step. The chunk that triggered it is encoded *as part of* the re-encode, so it is never encoded twice.

The re-encode is a **one-shot prefill**: the whole `[speaker_cache + fifo]` window goes through the layers in a single
parallel pass with the chunked-limited causal mask — numerically identical to streaming it in sub-chunks (positions are
relative), but at offline-encoder speed instead of many sequential passes.

### What it costs, what it approximates

- **Incremental step:** one chunk (~13 frames) through 17 layers + the head. Flat per step.
- **Compression step:** additionally one ~`(spkcache_len + fifo_len)`-frame prefill (~190 frames), ~once per period.

This is the causal `[-1, right_context]` approximation of the offline model (the same approximation the rest of the
streaming path uses): the encoder runs the trained non-causal weights causally, no retraining. On a real 2-speaker
clip it tracks the offline model to ~98% frame agreement.

### Config knobs (on `model.config`, all in 80 ms subsampled frames)

| Knob | Meaning |
| --- | --- |
| `chunk_len` | Frames committed per step. Smaller → lower latency, off the validated config. |
| `chunk_right_context` | Intra-chunk look-ahead of the causal mask (`right_context`). Adds `right_context * 80 ms` look-ahead latency. |
| `fifo_len` | Recent-context window. Must be **≥ `spkcache_len`** so the first compression has enough candidates. |
| `spkcache_len` | Long-term speaker-cache capacity. Sets how much speaker history the encoder + head attend to. |
| `spkcache_update_period` | Frames between full re-encodes (the `compression_period`). Larger → re-encode less often. |

Set these directly on `model.config` before calling `diarize_streaming_incremental`.
