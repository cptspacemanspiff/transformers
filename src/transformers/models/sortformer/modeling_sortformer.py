# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""PyTorch Sortformer speaker-diarization model."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import ModelOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ..parakeet.modeling_parakeet import ParakeetEncoder
from .configuration_sortformer import SortformerConfig


@dataclass
class SortformerModelOutput(ModelOutput):
    r"""
    Base class for [`SortformerModel`] outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
            Sequence of per-frame hidden states produced by the Transformer encoder on top of the FastConformer
            acoustic encoder.
        attention_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
            The (subsampled) frame-level attention mask, with `1` for valid frames and `0` for padding frames.
    """

    last_hidden_state: torch.FloatTensor = None
    attention_mask: torch.LongTensor | None = None


class SortformerAttention(nn.Module):
    """Multi-head self-attention matching NeMo's `MultiHeadAttention` (post-LN is applied by the encoder layer)."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads "
                f"({config.num_attention_heads})."
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = math.sqrt(self.head_dim)

        self.query_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_projection = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.layer_dropout = nn.Dropout(config.attention_layer_dropout)

    def _shape(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._shape(self.query_net(hidden_states), batch_size, seq_len)
        key = self._shape(self.key_net(hidden_states), batch_size, seq_len)
        value = self._shape(self.value_net(hidden_states), batch_size, seq_len)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scaling
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)

        output = self.out_projection(context)
        output = self.layer_dropout(output)
        return output


class SortformerFeedForward(nn.Module):
    """Position-wise feed-forward matching NeMo's `PositionWiseFF`."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SortformerEncoderLayer(nn.Module):
    """Post-LayerNorm Transformer encoder block (NeMo `TransformerEncoderBlock`, `pre_ln=False`)."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.self_attn = SortformerAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = SortformerFeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output = self.self_attn(hidden_states, attention_mask)
        hidden_states = self.layer_norm_1(hidden_states + attn_output)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm_2(hidden_states + ff_output)
        return hidden_states


class SortformerTransformerEncoder(nn.Module):
    """Stack of post-LN Transformer encoder layers. No final layer norm (pre_ln=False in NeMo)."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([SortformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class SortformerSpeakerHead(nn.Module):
    """Per-frame speaker-activity head (NeMo `forward_speaker_sigmoids`). Returns pre-sigmoid logits."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.head_dropout)
        self.act_fn = ACT2FN[config.hidden_act]
        self.first_hidden_to_hidden = nn.Linear(config.hidden_size, config.hidden_size)
        self.single_hidden_to_spks = nn.Linear(config.hidden_size, config.num_speakers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.act_fn(hidden_states))
        hidden_states = self.first_hidden_to_hidden(hidden_states)
        hidden_states = self.dropout(self.act_fn(hidden_states))
        logits = self.single_hidden_to_spks(hidden_states)
        return logits


@auto_docstring
class SortformerPreTrainedModel(PreTrainedModel):
    config_class = SortformerConfig
    base_model_prefix = "sortformer"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SortformerEncoderLayer", "ParakeetEncoderBlock"]
    # NOTE: weight init is inherited from `PreTrainedModel._init_weights`, which uses the guarded
    # `transformers.initialization` helpers (so already-loaded params are not re-initialized). Sortformer adds no
    # custom `nn.Parameter`s beyond standard `nn.Linear`/`nn.LayerNorm`, so no override is needed. Do NOT re-init with
    # raw `tensor.data.normal_()` — that bypasses the guard and clobbers weights loaded by `from_pretrained`.


@auto_docstring(
    custom_intro="""
    The bare Sortformer model: a FastConformer acoustic encoder followed by a post-LN Transformer encoder, producing
    per-frame hidden states.
    """
)
class SortformerModel(SortformerPreTrainedModel):
    def __init__(self, config: SortformerConfig):
        super().__init__(config)
        self.encoder = ParakeetEncoder(config.encoder_config)
        if config.encoder_config.hidden_size != config.hidden_size:
            self.encoder_proj = nn.Linear(config.encoder_config.hidden_size, config.hidden_size)
        else:
            self.encoder_proj = None
        self.transformer_encoder = SortformerTransformerEncoder(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> SortformerModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`):
            Log-mel filterbank features extracted from the raw waveform by [`SortformerFeatureExtractor`].
        """
        encoder_outputs = self.encoder(input_features, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        frame_mask = encoder_outputs.attention_mask

        if self.encoder_proj is not None:
            hidden_states = self.encoder_proj(hidden_states)

        additive_mask = None
        if frame_mask is not None:
            additive_mask = (1.0 - frame_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(
                hidden_states.dtype
            ).min

        hidden_states = self.transformer_encoder(hidden_states, additive_mask)

        return SortformerModelOutput(last_hidden_state=hidden_states, attention_mask=frame_mask)


@auto_docstring(
    custom_intro="""
    Sortformer model with a per-frame speaker-activity head on top (for end-to-end speaker diarization). Emits, for
    each output frame, an independent probability for each speaker being active (multi-label / sigmoid).
    """
)
class SortformerForAudioFrameClassification(SortformerPreTrainedModel):
    def __init__(self, config: SortformerConfig):
        super().__init__(config)
        self.sortformer = SortformerModel(config)
        self.speaker_head = SortformerSpeakerHead(config)
        self.num_speakers = config.num_speakers
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`):
            Log-mel filterbank features extracted from the raw waveform by [`SortformerFeatureExtractor`].
        labels (`torch.FloatTensor` of shape `(batch_size, num_output_frames, num_speakers)`, *optional*):
            Per-frame, per-speaker binary activity targets in `{0, 1}`. When provided, a `BCEWithLogitsLoss` is
            returned. Speaker probabilities are obtained from the returned `logits` via `logits.sigmoid()`.
        """
        outputs = self.sortformer(input_features, attention_mask=attention_mask)
        logits = self.speaker_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.to(logits.dtype))

        return TokenClassifierOutput(loss=loss, logits=logits)


__all__ = ["SortformerModel", "SortformerForAudioFrameClassification", "SortformerPreTrainedModel"]
