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

This implementation covers the **offline** (full-sequence) forward path. The original streaming inference with the
Arrival-Order Speaker Cache (AOSC) is not yet ported.

## Usage

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
