<!--Copyright 2026 The NVIDIA NeMo Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Sortformer

## Overview

Sortformer is an end-to-end neural speaker-diarization model introduced by NVIDIA NeMo in
[Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens](https://huggingface.co/papers/2409.06656).
It resolves the speaker-permutation problem by *sorting* speakers in order of their arrival time, which lets
diarization be trained with a simple per-frame binary objective and integrated directly with ASR.

**Model Architecture**

- **FastConformer encoder**: a linearly scalable Conformer that turns 128-bin log-mel features into subsampled
  (8×, i.e. 80 ms/frame) acoustic embeddings. This is the same encoder as [`ParakeetEncoder`], which Sortformer
  reuses.
- **Transformer encoder**: an 18-layer post-LayerNorm Transformer encoder operating on the (projected) acoustic
  embeddings.
- **Speaker-activity head**: a small MLP producing, for each output frame, an independent sigmoid probability for
  each of `num_speakers` speakers (multi-label). The model is exposed as
  [`SortformerForAudioFrameClassification`].

The model emits per-frame, per-speaker activity probabilities of shape `(batch_size, num_frames, num_speakers)`.
Probabilities are obtained from the returned `logits` via `logits.sigmoid()`.

Both the **offline** (full-sequence) forward path and the **streaming** inference path with the Arrival-Order Speaker
Cache (AOSC) are supported. Streaming reuses the same weights but processes the audio in fixed-size chunks, keeping a
bounded speaker cache so that memory and compute stay constant regardless of recording length.

## Usage

### Offline (full-sequence)

```python
import torch
from transformers import AutoFeatureExtractor, SortformerForAudioFrameClassification

model_id = "nvidia/diar_streaming_sortformer_4spk-v2-hf"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = SortformerForAudioFrameClassification.from_pretrained(model_id).eval()

# `audio` is a 16 kHz mono waveform (numpy array or list of floats)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# per-frame, per-speaker activity probabilities; each frame spans 80 ms
speaker_probs = logits.sigmoid()  # (batch, num_frames, num_speakers)
speaker_active = speaker_probs > 0.5
```

### Streaming (Arrival-Order Speaker Cache)

[`~SortformerForAudioFrameClassification.diarize_streaming`] runs the chunked loop internally and returns the
per-frame speaker probabilities directly (already `sigmoid`-activated). Unlike the offline path, the streaming path
should be fed features extracted **without** waveform peak-normalization.

```python
with torch.no_grad():
    speaker_probs = model.diarize_streaming(inputs.input_features, attention_mask=inputs.attention_mask)
```

For true incremental / real-time use, drive the steps yourself with the explicit streaming state (the analog of a
generation cache), which is created by [`~SortformerForAudioFrameClassification.init_streaming_state`] and threaded
through each [`~SortformerForAudioFrameClassification.streaming_step`] call. The chunking and cache hyperparameters
(`chunk_len`, `spkcache_len`, `fifo_len`, …) are configurable on [`SortformerConfig`].

A runnable example that plots the diarization result (with an optional `--streaming` flag) lives at
`examples/pytorch/speaker-diarization/run_sortformer_diarization.py`.

Checkpoints in the original NeMo `.nemo` format can be converted with
`src/transformers/models/sortformer/convert_sortformer_nemo_to_hf.py`.

## SortformerConfig

[[autodoc]] SortformerConfig

## SortformerFeatureExtractor

[[autodoc]] SortformerFeatureExtractor

## SortformerModel

[[autodoc]] SortformerModel
    - forward

## SortformerForAudioFrameClassification

[[autodoc]] SortformerForAudioFrameClassification
    - forward
    - diarize_streaming
    - streaming_step
    - init_streaming_state

## SortformerStreamingState

[[autodoc]] SortformerStreamingState
