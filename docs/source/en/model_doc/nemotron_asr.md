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
*This model was contributed to Hugging Face Transformers on 2026-06-19.*

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# Nemotron ASR

## Overview

[Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) is a multilingual, streaming
automatic speech recognition model from NVIDIA NeMo. It pairs a **cache-aware streaming [Fast Conformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer)
encoder** with an **RNN-T (RNN Transducer)** decoder and conditions transcription on a **language-ID prompt**,
supporting 40 language-locales from a single model.

**Model Architecture**

- **Cache-Aware Fast Conformer Encoder** ([`NemotronAsrEncoder`]): a streaming variant of the Parakeet/Fast Conformer
  encoder. Unlike the offline Parakeet encoder, its convolutions are *causal* (causal downsampling and a causal
  depthwise convolution) and its self-attention uses a *limited, chunked context window* (`att_context_size`,
  `att_context_style`). The `[left, right]` context (in 80 ms frames) trades latency for accuracy; e.g. `[56, 13]`
  corresponds to a 1.12 s look-ahead.
- **Language-ID prompt fusion**: a one-hot language vector (width `num_prompts`, selected by `prompt_indices`) is
  broadcast across time, concatenated to the encoder output, and projected back to the hidden size by a small MLP
  (`prompt_kernel`) before the joint network.
- **RNN-T decoder** ([`NemotronAsrForRNNT`]): an LSTM prediction network plus a joint network, decoded greedily (a
  blank emission advances the encoder frame by one, a non-blank emission stays on the same frame).

This model transcribes with punctuation and capitalization, and (with `target_lang="auto"`) can detect the spoken
language and append its tag (e.g. `<en-US>`) after the terminal punctuation.

This model was contributed by the community. The original code can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Usage

```python
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, NemotronAsrForRNNT

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = NemotronAsrForRNNT.from_pretrained(model_id).eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

inputs = processor(ds[0]["audio"]["array"], sampling_rate=16000, target_lang="en-US", return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs)
print(processor.batch_decode(output.sequences, skip_special_tokens=True))
```

Convert an original NeMo `.nemo` checkpoint with
`src/transformers/models/nemotron_asr/convert_nemotron_asr_nemo_to_hf.py`.

### Cache-aware streaming

The model also supports genuine cache-aware streaming: each step consumes only a new audio chunk and reuses cached
encoder context (per-layer attention and convolution caches), so there is no overlapping recomputation. The latency
is set by `att_context_size = [left, right]` (in 80 ms frames); e.g. `[56, 0]` is 80 ms and `[56, 13]` is 1.12 s.

```python
import torch
from transformers import AutoProcessor, NemotronAsrForRNNT

model = NemotronAsrForRNNT.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b").eval()
processor = AutoProcessor.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")

inputs = processor(audio_array, sampling_rate=16000, target_lang="en-US", return_tensors="pt")
# convenience driver: chunk internally and return emitted token ids per batch element
token_ids = model.transcribe_stream(inputs["input_features"], prompt_indices=inputs["prompt_indices"])
print(processor.tokenizer.decode(token_ids[0], skip_special_tokens=True))
```

For a real per-chunk stream, drive it manually with [`~NemotronAsrForRNNT.init_streaming_state`] and
[`~NemotronAsrForRNNT.streaming_step`], threading the returned `NemotronAsrStreamingCache` from one chunk to the
next. The streaming output is numerically identical (frame-for-frame) to the offline forward.

## NemotronAsrConfig

[[autodoc]] NemotronAsrConfig

## NemotronAsrEncoderConfig

[[autodoc]] NemotronAsrEncoderConfig

## NemotronAsrFeatureExtractor

[[autodoc]] NemotronAsrFeatureExtractor

## NemotronAsrTokenizer

[[autodoc]] NemotronAsrTokenizer

## NemotronAsrProcessor

[[autodoc]] NemotronAsrProcessor

## NemotronAsrEncoder

[[autodoc]] NemotronAsrEncoder
    - forward

## NemotronAsrForRNNT

[[autodoc]] NemotronAsrForRNNT
    - forward
    - generate
