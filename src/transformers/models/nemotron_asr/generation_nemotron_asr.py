# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch

from ..parakeet.generation_parakeet import ParakeetRNNTDecoderCache, ParakeetRNNTGenerationMixin


class NemotronAsrRNNTDecoderCache(ParakeetRNNTDecoderCache):
    pass


@dataclass
class NemotronAsrStreamingCache:
    """Mutable state carried across [`NemotronAsrForRNNT.streaming_step`] calls for cache-aware streaming.

    It holds the per-layer encoder caches (the attention left-context and the convolution left-context that NeMo's
    `cache_last_channel` / `cache_last_time` store), the subsampling left-context mel frames, and the RNN-T decoder
    state (LSTM hidden/cell + last emitted token), so that each streaming step processes only the new audio chunk.

    Attributes:
        last_channel (`list[torch.FloatTensor]`): per-layer attention input cache, each `(batch, cache_len, hidden)`.
        last_time (`list[torch.FloatTensor]`): per-layer causal-conv cache, each `(batch, hidden, kernel - 1)`.
        last_channel_len (`int`): number of valid (filled) frames currently in `last_channel`.
        pre_encode (`torch.FloatTensor`): trailing mel frames `(batch, pre_encode_frames, num_mel_bins)` carried over
            to give the causal subsampling its left context on the next chunk.
        decoder_cache (`NemotronAsrRNNTDecoderCache`): RNN-T prediction-network LSTM state.
        last_token (`torch.LongTensor`): the last non-blank token emitted per batch element, `(batch, 1)`.
        finished (`bool`): set once a terminating step has been processed.
    """

    last_channel: list = field(default_factory=list)
    last_time: list = field(default_factory=list)
    last_channel_len: int = 0
    pre_encode: torch.FloatTensor | None = None
    decoder_cache: NemotronAsrRNNTDecoderCache | None = None
    last_token: torch.LongTensor | None = None
    finished: bool = False


class NemotronAsrRNNTGenerationMixin(ParakeetRNNTGenerationMixin):
    """RNN-T greedy generation for Nemotron 3.5 ASR.

    Identical to [`ParakeetRNNTGenerationMixin`] except that the language-ID `prompt_indices` are threaded into the
    single encoder pass done at the start of generation (the prompt fusion happens inside `get_audio_features`).
    """

    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None):
        # mirror ParakeetRNNTGenerationMixin but forward prompt_indices to the encoder
        from ...generation import GenerationMixin

        inputs, input_name, model_kwargs = GenerationMixin._prepare_model_inputs(
            self, inputs, bos_token_id, model_kwargs
        )

        encoder_outputs = self.get_audio_features(
            input_features=inputs,
            attention_mask=model_kwargs.get("attention_mask", None),
            prompt_indices=model_kwargs.get("prompt_indices", None),
            output_attention_mask=True,
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        if encoder_outputs.attention_mask is not None:
            encoder_valid_lengths = encoder_outputs.attention_mask.sum(-1)
        else:
            batch_size = encoder_outputs.last_hidden_state.shape[0]
            encoder_valid_lengths = torch.full(
                (batch_size,),
                encoder_outputs.last_hidden_state.shape[1],
                dtype=torch.long,
                device=encoder_outputs.last_hidden_state.device,
            )
        model_kwargs["encoder_valid_lengths"] = encoder_valid_lengths
        model_kwargs["encoder_frame_idxs"] = torch.zeros(inputs.shape[0], device=inputs.device, dtype=torch.long)

        # prompt_indices has done its job (fused into encoder_outputs); drop it so the decoder forward doesn't see it
        model_kwargs.pop("prompt_indices", None)

        return inputs, input_name, model_kwargs


__all__ = ["NemotronAsrRNNTDecoderCache", "NemotronAsrRNNTGenerationMixin", "NemotronAsrStreamingCache"]
