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
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import ModelOutput, TokenClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.output_capturing import capture_outputs
from ..parakeet.modeling_parakeet import ParakeetEncoder
from .configuration_sortformer import SortformerConfig


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@dataclass
class SortformerStreamingState(ModelOutput):
    r"""
    Mutable state carried across [`SortformerForAudioFrameClassification.streaming_step`] calls (the Arrival-Order
    Speaker Cache). This is the streaming analog of a generation cache: it is created by
    [`~SortformerForAudioFrameClassification.init_streaming_state`], threaded through each step, and updated in place.

    Args:
        spkcache (`torch.FloatTensor` of shape `(batch_size, cache_len, fc_d_model)`):
            Speaker-cache embeddings accumulated from the start of the stream (compressed to at most `spkcache_len`).
        spkcache_preds (`torch.FloatTensor` of shape `(batch_size, cache_len, num_speakers)`, *optional*):
            Speaker probabilities associated with the speaker-cache frames (populated after the first compression).
        fifo (`torch.FloatTensor` of shape `(batch_size, fifo_len, fc_d_model)`):
            FIFO queue of the most recent embeddings awaiting promotion into the speaker cache.
        fifo_preds (`torch.FloatTensor` of shape `(batch_size, fifo_len, num_speakers)`, *optional*):
            Speaker probabilities associated with the FIFO-queue frames.
        mean_sil_emb (`torch.FloatTensor` of shape `(batch_size, fc_d_model)`):
            Running mean embedding of frames classified as silence, used to fill disabled speaker-cache slots.
        n_sil_frames (`torch.LongTensor` of shape `(batch_size,)`):
            Running count of silence frames accumulated into `mean_sil_emb`.
    """

    spkcache: torch.FloatTensor = None
    spkcache_preds: torch.FloatTensor | None = None
    fifo: torch.FloatTensor = None
    fifo_preds: torch.FloatTensor | None = None
    mean_sil_emb: torch.FloatTensor = None
    n_sil_frames: torch.LongTensor = None


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
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Per-layer hidden states of the Transformer encoder (one per layer plus the input embeddings).
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Per-layer attention weights of the Transformer encoder.
    """

    last_hidden_state: torch.FloatTensor = None
    attention_mask: torch.LongTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class SortformerAttention(nn.Module):
    """Multi-head self-attention matching NeMo's `MultiHeadAttention` (post-LN is applied by the encoder layer).

    Routes through the standard `ALL_ATTENTION_FUNCTIONS` interface, so it supports eager / sdpa / flash backends via
    `config._attn_implementation` and is `torch.compile`-friendly.
    """

    def __init__(self, config: SortformerConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads "
                f"({config.num_attention_heads})."
            )
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1  # no grouped-query attention
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.query_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_projection = nn.Linear(config.hidden_size, config.hidden_size)

        self.layer_dropout = nn.Dropout(config.attention_layer_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query = self.query_net(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.key_net(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.value_net(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        output = self.out_projection(attn_output)
        output = self.layer_dropout(output)
        return output, attn_weights


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(hidden_states, attention_mask, **kwargs)
        hidden_states = self.layer_norm_1(hidden_states + attn_output)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm_2(hidden_states + ff_output)
        return hidden_states


class SortformerTransformerEncoder(nn.Module):
    """Stack of post-LN Transformer encoder layers. No final layer norm (pre_ln=False in NeMo)."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([SortformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, **kwargs)
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
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": SortformerEncoderLayer,
        "attentions": SortformerAttention,
    }
    # NOTE: weight init is inherited from `PreTrainedModel._init_weights`, which uses the guarded
    # `transformers.initialization` helpers (so already-loaded params are not re-initialized). Sortformer adds no
    # custom `nn.Parameter`s beyond standard `nn.Linear`/`nn.LayerNorm`, so no override is needed. Do NOT re-init with
    # raw `tensor.data.normal_()` — that bypasses the guard and clobbers weights loaded by `from_pretrained`.


# ----------------------------------------------------------------------------------------------------------------------
# Cache-aware streaming of the FastConformer encoder layers.
#
# The same FastConformer (Parakeet) layers are run in one of two modes by `SortformerModel._conformer_forward`:
#   * offline (`cache is None`): the standard `ParakeetEncoderBlock.forward` over the whole window (optionally with the
#     chunked-limited causal mask). This is the offline / full-re-encode path and is numerically unchanged.
#   * streaming (`cache is not None`): the block math is inlined, but attention reads/writes a growing projected K/V
#     cache and the depthwise convolution is made causal with a left-context cache. Each query frame attends to the
#     whole past plus `right_context` look-ahead frames within its own `(right_context + 1)`-frame chunk.
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class SortformerStreamingConformerState:
    """Per-layer cache for cache-aware streaming of the FastConformer encoder layers.

    The attention cache stores the already-**projected** keys/values (a true KV cache): each step only projects the
    new chunk's frames and appends them, instead of re-projecting the whole history. Because `k_proj`/`v_proj` are
    position-wise linear, this is numerically identical to re-projecting `[cache, new]` every step.

    Attributes:
        key_cache (`list[torch.FloatTensor]`): per-layer projected keys `(batch, num_heads, cache_len, head_dim)`,
            growing unboundedly so the model attends to the entire past.
        value_cache (`list[torch.FloatTensor]`): per-layer projected values `(batch, num_heads, cache_len, head_dim)`.
        last_time (`list[torch.FloatTensor]`): per-layer causal-conv left context `(batch, d_model, kernel - 1)`.
        offset (`int`): absolute index (in subsampled frames) of the next chunk, used for chunk-boundary alignment.
    """

    key_cache: list
    value_cache: list
    last_time: list
    offset: int = 0


def _conformer_streaming_attention(
    attention, hidden_states, position_embeddings, key_cache, value_cache, attention_mask
):
    """Relative-position self-attention with a projected key/value cache, on a single streaming chunk.

    `hidden_states` are the new query frames; `key_cache`/`value_cache` hold the projected keys/values from all past
    frames `(batch, num_heads, cache_len, head_dim)`. Only the new frames are projected and appended. Returns the
    attention output plus the updated key/value caches.
    """
    batch_size, query_length = hidden_states.shape[:2]
    num_heads, head_dim = attention.config.num_attention_heads, attention.head_dim
    hidden_shape = (batch_size, query_length, num_heads, head_dim)

    query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = torch.cat([key_cache, attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)], dim=2)
    value_states = torch.cat([value_cache, attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)], dim=2)
    kv_length = key_states.shape[2]

    query_with_bias_u = query_states + attention.bias_u.view(1, num_heads, 1, head_dim)
    query_with_bias_v = query_states + attention.bias_v.view(1, num_heads, 1, head_dim)

    relative_key_states = attention.relative_k_proj(position_embeddings).view(batch_size, -1, num_heads, head_dim)
    matrix_bd = query_with_bias_v @ relative_key_states.permute(0, 2, 3, 1)
    matrix_bd = attention._rel_shift(matrix_bd)[..., :kv_length]
    matrix_ac = query_with_bias_u @ key_states.transpose(-2, -1)

    scores = (matrix_ac + matrix_bd) * attention.scaling
    scores = scores.masked_fill(attention_mask[:, None].logical_not(), torch.finfo(scores.dtype).min)
    attn_weights = scores.softmax(dim=-1)
    attn_output = (attn_weights @ value_states).transpose(1, 2).reshape(batch_size, query_length, -1)
    return attention.o_proj(attn_output), key_states, value_states


def _conformer_streaming_conv(conv_module, hidden_states, cache, left_padding):
    """Causal depthwise convolution on one streaming chunk, using `cache` (the previous `kernel - 1` input frames).

    Runs the offline `ParakeetEncoderConvolutionModule` weights with left-only padding (the cache) instead of the
    symmetric padding the module normally applies.
    """
    hidden_states = hidden_states.transpose(1, 2)  # (B, C, n)
    hidden_states = conv_module.pointwise_conv1(hidden_states)
    hidden_states = nn.functional.glu(hidden_states, dim=1)

    combined = torch.cat([cache, hidden_states], dim=-1)
    new_cache = combined[:, :, -left_padding:]
    hidden_states = nn.functional.conv1d(
        combined,
        conv_module.depthwise_conv.weight,
        conv_module.depthwise_conv.bias,
        groups=conv_module.depthwise_conv.groups,
    )
    hidden_states = conv_module.norm(hidden_states)  # BatchNorm1d, in eval uses running statistics
    hidden_states = conv_module.activation(hidden_states)
    hidden_states = conv_module.pointwise_conv2(hidden_states)
    return hidden_states.transpose(1, 2), new_cache


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

    @auto_docstring
    @capture_outputs
    @can_return_tuple
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
        hidden_states, frame_mask, _ = self._conformer_forward(input_features, attention_mask=attention_mask)

        if self.encoder_proj is not None:
            hidden_states = self.encoder_proj(hidden_states)

        additive_mask = None
        if frame_mask is not None:
            additive_mask = (1.0 - frame_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(
                hidden_states.dtype
            ).min

        hidden_states = self.transformer_encoder(hidden_states, additive_mask)

        return SortformerModelOutput(last_hidden_state=hidden_states, attention_mask=frame_mask)

    @staticmethod
    def _length_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        arange = torch.arange(max_length, device=lengths.device)
        return arange.expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)

    def _conformer_forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: SortformerStreamingConformerState | None = None,
        causal: bool = False,
        right_context: int = 7,
    ):
        """Single FastConformer-encoder forward shared by the offline and cache-aware streaming paths.

        The convolutional subsampling is always run once up front (its weights are non-causal). The `num_hidden_layers`
        conformer layers are then run in one of two modes:

        * `cache is None` (offline / full re-encode): the standard `ParakeetEncoderBlock.forward` over the whole window,
          bidirectional when `causal=False`, or restricted to the chunked-limited `[-1, right_context]` mask when
          `causal=True`. This is numerically identical to `self.encoder(...)` followed by no caching.
        * `cache is not None` (cache-aware streaming): the block math is inlined, reading/writing the per-layer projected
          K/V cache and the causal-conv left-context cache in `cache`. Each query frame attends to the whole past plus
          `right_context` look-ahead frames within its own `(right_context + 1)`-frame chunk.

        Returns `(hidden_states, output_mask, updated_cache)` where `output_mask` is the subsampled frame mask (or
        `None` when no input mask was given) and `updated_cache` is the mutated `cache` (or `None` in offline mode).
        """
        encoder = self.encoder

        if cache is None:
            hidden_states = encoder.subsampling(input_features, attention_mask)
            hidden_states = hidden_states * encoder.input_scale
            position_embeddings = encoder.encode_positions(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=encoder.dropout, training=self.training)
            position_embeddings = nn.functional.dropout(
                position_embeddings, p=encoder.dropout_positions, training=self.training
            )

            seq_len = hidden_states.shape[1]
            output_mask = None
            attn_keep = None
            if attention_mask is not None:
                output_mask = encoder._get_output_attention_mask(attention_mask, target_length=seq_len)
                attn_keep = output_mask.unsqueeze(1).expand(-1, seq_len, -1)
                attn_keep = attn_keep & attn_keep.transpose(1, 2)
            if causal:
                block = torch.arange(seq_len, device=hidden_states.device) // (right_context + 1)
                causal_keep = (block[:, None] >= block[None, :]).unsqueeze(0)  # query block >= key block
                attn_keep = causal_keep if attn_keep is None else attn_keep & causal_keep
            layer_mask = attn_keep.unsqueeze(1) if attn_keep is not None else None

            for encoder_layer in encoder.layers:
                hidden_states = encoder_layer(
                    hidden_states, attention_mask=layer_mask, position_embeddings=position_embeddings
                )
            # Match `ParakeetEncoder.forward`, which returns the subsampled frame mask as an int tensor.
            return hidden_states, (output_mask.int() if output_mask is not None else None), None

        # Cache-aware streaming path: subsampling has already been applied by the caller (which feeds chunks of
        # already-subsampled, *unscaled* embeddings as `input_features`). `attention_mask` is ignored (synchronous,
        # batch-uniform). Scaling is applied here so the caller mirrors offline subsampling exactly.
        hidden_states = self._conformer_streaming_step(input_features * encoder.input_scale, cache, right_context)
        return hidden_states, None, cache

    def init_streaming_conformer_state(
        self, batch_size: int = 1, device=None, dtype=None
    ) -> SortformerStreamingConformerState:
        """Create an empty [`SortformerStreamingConformerState`] for cache-aware streaming of the encoder layers."""
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        encoder_config = self.config.encoder_config
        hidden_size = encoder_config.hidden_size
        num_heads = encoder_config.num_attention_heads
        head_dim = hidden_size // num_heads
        conv_cache = encoder_config.conv_kernel_size - 1
        num_layers = encoder_config.num_hidden_layers
        return SortformerStreamingConformerState(
            key_cache=[
                torch.zeros(batch_size, num_heads, 0, head_dim, device=device, dtype=dtype) for _ in range(num_layers)
            ],
            value_cache=[
                torch.zeros(batch_size, num_heads, 0, head_dim, device=device, dtype=dtype) for _ in range(num_layers)
            ],
            last_time=[
                torch.zeros(batch_size, hidden_size, conv_cache, device=device, dtype=dtype) for _ in range(num_layers)
            ],
            offset=0,
        )

    def _conformer_streaming_step(
        self, scaled_chunk: torch.Tensor, state: SortformerStreamingConformerState, right_context: int = 7
    ) -> torch.Tensor:
        """Run the FastConformer layers on one chunk of already-subsampled-and-scaled embeddings, cache-aware.

        Each query frame attends to the whole history (growing K/V cache) and to look-ahead frames within its own chunk
        (`chunked_limited` with unlimited left, right context `right_context`); convolutions are causal. Updates `state`
        in place and returns the chunk's encoder embeddings `(batch, n, d_model)`.
        """
        encoder = self.encoder
        chunk_size = right_context + 1
        cache_len = state.key_cache[0].shape[2]
        n = scaled_chunk.shape[1]
        kv_length = cache_len + n

        position_embeddings = encoder.encode_positions(
            scaled_chunk.new_zeros(scaled_chunk.shape[0], kv_length, scaled_chunk.shape[2])
        )
        # chunked-limited keep-mask with unlimited left: a query attends keys in its own chunk (look-ahead up to the
        # chunk boundary) and all earlier chunks, never a later chunk.
        absolute = torch.arange(state.offset - cache_len, state.offset + n, device=scaled_chunk.device)
        chunk_idx = torch.div(absolute, chunk_size, rounding_mode="trunc")
        diff = chunk_idx[cache_len:, None] - chunk_idx[None, :]
        attention_mask = (diff >= 0).unsqueeze(0)

        left_padding = self.config.encoder_config.conv_kernel_size - 1
        hidden_states = scaled_chunk
        new_key, new_value, new_time = [], [], []
        for i, layer in enumerate(encoder.layers):
            hidden_states = hidden_states + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(hidden_states))
            normed = layer.norm_self_att(hidden_states)
            attn_output, key_states, value_states = _conformer_streaming_attention(
                layer.self_attn, normed, position_embeddings, state.key_cache[i], state.value_cache[i], attention_mask
            )
            hidden_states = hidden_states + attn_output
            conv_output, conv_cache = _conformer_streaming_conv(
                layer.conv, layer.norm_conv(hidden_states), state.last_time[i], left_padding
            )
            hidden_states = hidden_states + conv_output
            hidden_states = hidden_states + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(hidden_states))
            hidden_states = layer.norm_out(hidden_states)
            new_key.append(key_states)
            new_value.append(value_states)
            new_time.append(conv_cache)

        state.key_cache, state.value_cache, state.last_time, state.offset = (
            new_key,
            new_value,
            new_time,
            state.offset + n,
        )
        return hidden_states

    def streaming_encode(
        self, input_features: torch.Tensor, right_context: int = 7, chunk_size: int | None = None
    ) -> torch.Tensor:
        """Cache-aware streaming of the FastConformer encoder over `input_features` (whole-utterance convenience).

        The convolutional subsampling is run once up front (it is cheap and its weights are non-causal); the conformer
        layers are then streamed in `right_context + 1`-frame chunks (or `chunk_size` frames) with growing attention
        caches. Returns the encoder output `(batch, num_subsampled_frames, d_model)`, the streaming analog of
        `self.encoder(...)`.
        """
        encoder = self.encoder
        subsampled = encoder.subsampling(input_features)
        step = chunk_size if chunk_size is not None else right_context + 1
        state = self.init_streaming_conformer_state(subsampled.shape[0], subsampled.device, subsampled.dtype)
        outputs, pos, total = [], 0, subsampled.shape[1]
        while pos < total:
            end = min(pos + step, total)
            hidden_states, _, state = self._conformer_forward(
                subsampled[:, pos:end], cache=state, right_context=right_context
            )
            outputs.append(hidden_states)
            pos = end
        return torch.cat(outputs, dim=1)


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
        outputs = self.sortformer(input_features, attention_mask=attention_mask, **kwargs)
        logits = self.speaker_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.to(logits.dtype))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def streaming_diarize(
        self, input_features: torch.Tensor, right_context: int = 7, chunk_size: int | None = None
    ) -> torch.Tensor:
        """Diarize with the cache-aware *streaming conformer* encoder (see [`SortformerModel.streaming_encode`]).

        The FastConformer runs as a low-latency cache-aware stream (causal convs + growing attention cache + bounded
        `right_context` look-ahead); the Transformer encoder and speaker head then run over the resulting embeddings.
        Returns per-frame speaker-activity `logits` of shape `(batch, num_frames, num_speakers)` (apply `.sigmoid()`).
        """
        hidden_states = self.sortformer.streaming_encode(input_features, right_context, chunk_size)
        if self.sortformer.encoder_proj is not None:
            hidden_states = self.sortformer.encoder_proj(hidden_states)
        hidden_states = self.sortformer.transformer_encoder(hidden_states, None)
        return self.speaker_head(hidden_states)

    # ------------------------------------------------------------------------------------------------------------------
    # Streaming (Arrival-Order Speaker Cache) inference.
    #
    # This is a faithful port of NeMo's *synchronous* streaming path (`SortformerModules.streaming_update` /
    # `_compress_spkcache` and `SortformerEncLabelModel.forward_streaming`), used for offline/batch streaming
    # evaluation. It is an inference-only API: the training-only code paths (random speaker permutation, score noise,
    # causal-attention modification) are intentionally omitted. Unlike the offline `forward`, the streaming path does
    # **not** peak-normalize the waveform, so features should be extracted directly (without peak normalization).
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _fc_d_model(self) -> int:
        return self.config.encoder_config.hidden_size

    @property
    def _subsampling_factor(self) -> int:
        return self.config.encoder_config.subsampling_factor

    def init_streaming_state(
        self, batch_size: int = 1, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> SortformerStreamingState:
        """
        Create an empty [`SortformerStreamingState`] for synchronous streaming inference.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Batch size of the stream.
            device (`torch.device`, *optional*):
                Device for the state tensors. Defaults to the model's device.
            dtype (`torch.dtype`, *optional*):
                Floating dtype for the state tensors. Defaults to the model's dtype.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        fc_d_model = self._fc_d_model
        return SortformerStreamingState(
            spkcache=torch.zeros((batch_size, 0, fc_d_model), device=device, dtype=dtype),
            spkcache_preds=None,
            fifo=torch.zeros((batch_size, 0, fc_d_model), device=device, dtype=dtype),
            fifo_preds=None,
            mean_sil_emb=torch.zeros((batch_size, fc_d_model), device=device, dtype=dtype),
            n_sil_frames=torch.zeros((batch_size,), dtype=torch.long, device=device),
        )

    @staticmethod
    def _length_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        arange = torch.arange(max_length, device=lengths.device)
        return arange.expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)

    def _run_conformer_on_embeddings(
        self, embeddings: torch.Tensor, lengths: torch.Tensor, causal: bool = False, right_context: int = 7
    ):
        """
        Run the FastConformer encoder layers on already-subsampled `[spkcache, fifo, chunk]` embeddings, bypassing the
        convolutional pre-encode (subsampling). Mirrors the post-subsampling part of `ParakeetEncoder.forward`.

        With `causal=False` (default) the window is attended bidirectionally (this is NeMo's synchronous
        `forward_streaming`). With `causal=True` the chunked-limited `[-1, right_context]` mask is applied instead, so
        each frame attends only to the past plus `right_context` look-ahead within its own `(right_context + 1)`-frame
        block -- the from-scratch causal equivalent that the incremental path approximates with a cached re-encode.
        """
        encoder = self.sortformer.encoder
        hidden_states = embeddings * encoder.input_scale
        position_embeddings = encoder.encode_positions(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=encoder.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=encoder.dropout_positions, training=self.training
        )

        seq_len = hidden_states.shape[1]
        output_mask = self._length_to_mask(lengths, seq_len)
        attention_mask = output_mask.unsqueeze(1).expand(-1, seq_len, -1)
        attention_mask = attention_mask & attention_mask.transpose(1, 2)
        if causal:
            block = torch.arange(seq_len, device=hidden_states.device) // (right_context + 1)
            attention_mask = attention_mask & (block[:, None] >= block[None, :])  # query block >= key block
        attention_mask = attention_mask.unsqueeze(1)

        for encoder_layer in encoder.layers:
            hidden_states = encoder_layer(
                hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings
            )
        return hidden_states, output_mask

    def _forward_infer(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Project, run the Transformer encoder, and apply the speaker head, returning sigmoid probabilities."""
        hidden_states = embeddings
        if self.sortformer.encoder_proj is not None:
            hidden_states = self.sortformer.encoder_proj(hidden_states)

        additive_mask = None
        if lengths is not None:
            frame_mask = self._length_to_mask(lengths, hidden_states.shape[1])
            additive_mask = (1.0 - frame_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(
                hidden_states.dtype
            ).min

        hidden_states = self.sortformer.transformer_encoder(hidden_states, additive_mask)
        logits = self.speaker_head(hidden_states)
        return logits.sigmoid()

    @staticmethod
    def _apply_mask_to_preds(preds: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, n_frames, n_spk = preds.shape
        preds_mask = torch.arange(n_frames, device=preds.device).view(1, -1, 1)
        preds_mask = preds_mask.expand(batch_size, -1, n_spk) < lengths.view(-1, 1, 1)
        return torch.where(preds_mask, preds, torch.zeros((), device=preds.device, dtype=preds.dtype))

    def _get_silence_profile(self, mean_sil_emb, n_sil_frames, emb_seq, preds):
        is_sil = preds.sum(dim=2) < self.config.sil_threshold
        sil_count = is_sil.sum(dim=1)
        if not (sil_count > 0).any():
            return mean_sil_emb, n_sil_frames
        sil_emb_sum = torch.sum(emb_seq * is_sil.unsqueeze(-1), dim=1)
        upd_n_sil_frames = n_sil_frames + sil_count
        old_sil_emb_sum = mean_sil_emb * n_sil_frames.unsqueeze(1)
        upd_mean_sil_emb = (old_sil_emb_sum + sil_emb_sum) / torch.clamp(upd_n_sil_frames.unsqueeze(1), min=1)
        return upd_mean_sil_emb, upd_n_sil_frames

    def _get_log_pred_scores(self, preds: torch.Tensor) -> torch.Tensor:
        n_spk = self.config.num_speakers
        log_probs = torch.log(torch.clamp(preds, min=self.config.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.config.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, n_spk)
        return log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk: int) -> torch.Tensor:
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float("-inf"), device=scores.device))
        is_pos = scores > 0
        is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        scores = torch.where(is_nonpos_replace, torch.tensor(float("-inf"), device=scores.device), scores)
        return scores

    @staticmethod
    def _boost_topk_scores(scores, n_boost_per_spk: int, scale_factor: float = 1.0, offset: float = 0.5):
        if n_boost_per_spk <= 0:
            return scores
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
        batch_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1).unsqueeze(2)
        speaker_indices = torch.arange(n_spk, device=scores.device).unsqueeze(0).unsqueeze(0)
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor * math.log(offset)
        return scores

    def _get_topk_indices(self, scores: torch.Tensor):
        batch_size, n_frames, _ = scores.shape
        spkcache_len = self.config.spkcache_len
        n_frames_no_sil = n_frames - self.config.spkcache_sil_frames_per_spk
        scores_flatten = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(scores_flatten, spkcache_len, dim=1, sorted=False)
        valid_topk_mask = topk_values != float("-inf")
        topk_indices = torch.where(
            valid_topk_mask, topk_indices, torch.tensor(self.config.max_index, device=scores.device)
        )
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)
        is_disabled = topk_indices_sorted == self.config.max_index
        topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames)
        is_disabled += topk_indices_sorted >= n_frames_no_sil
        topk_indices_sorted[is_disabled] = 0
        return topk_indices_sorted, is_disabled

    def _gather_spkcache_and_preds(self, emb_seq, preds, topk_indices, is_disabled, mean_sil_emb):
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        indices_expanded_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        emb_seq_gathered = torch.gather(emb_seq, 1, indices_expanded_emb)
        mean_sil_emb_expanded = mean_sil_emb.unsqueeze(1).expand(-1, self.config.spkcache_len, -1)
        emb_seq_gathered = torch.where(is_disabled.unsqueeze(-1), mean_sil_emb_expanded, emb_seq_gathered)

        indices_expanded_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_gathered = torch.gather(preds, 1, indices_expanded_spk)
        preds_gathered = torch.where(
            is_disabled.unsqueeze(-1), torch.zeros((), device=preds.device, dtype=preds.dtype), preds_gathered
        )
        return emb_seq_gathered, preds_gathered

    def _spkcache_topk_indices(self, preds):
        """Score `preds` (salience) and return the `(topk_indices, is_disabled)` selecting `spkcache_len` frames.

        Factored out of [`_compress_spkcache`] so the incremental path can gather frozen K/V (not just embeddings) by
        the same indices. Candidate layout is `[old_spkcache, newer_frames]`, matching the `scores_boost_latest` slice.
        """
        batch_size, n_frames, n_spk = preds.shape
        spkcache_len_per_spk = self.config.spkcache_len // n_spk - self.config.spkcache_sil_frames_per_spk
        strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.config.strong_boost_rate)
        weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.config.weak_boost_rate)
        min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * self.config.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)

        if self.config.scores_boost_latest > 0:
            scores[:, self.config.spkcache_len :, :] += self.config.scores_boost_latest

        scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)

        if self.config.spkcache_sil_frames_per_spk > 0:
            pad = torch.full(
                (batch_size, self.config.spkcache_sil_frames_per_spk, n_spk), float("inf"), device=scores.device
            )
            scores = torch.cat([scores, pad], dim=1)

        return self._get_topk_indices(scores)

    def _compress_spkcache(self, emb_seq, preds, mean_sil_emb):
        """Compress the speaker cache to `spkcache_len` most-informative frames (inference-only; no permutation)."""
        topk_indices, is_disabled = self._spkcache_topk_indices(preds)
        return self._gather_spkcache_and_preds(emb_seq, preds, topk_indices, is_disabled, mean_sil_emb)

    def _streaming_update(self, state, chunk, preds, lc: int = 0, rc: int = 0):
        """Synchronous speaker-cache / FIFO update (NeMo `SortformerModules.streaming_update`)."""
        spkcache_len = state.spkcache.shape[1]
        fifo_len = state.fifo.shape[1]
        chunk_len = chunk.shape[1] - lc - rc

        state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        chunk = chunk[:, lc : chunk_len + lc]
        chunk_preds = preds[:, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]

        state.fifo = torch.cat([state.fifo, chunk], dim=1)
        state.fifo_preds = torch.cat([state.fifo_preds, chunk_preds], dim=1)

        if fifo_len + chunk_len > self.config.fifo_len:
            pop_out_len = self.config.spkcache_update_period
            pop_out_len = max(pop_out_len, chunk_len - self.config.fifo_len + fifo_len)
            pop_out_len = min(pop_out_len, fifo_len + chunk_len)

            pop_out_embs = state.fifo[:, :pop_out_len]
            pop_out_preds = state.fifo_preds[:, :pop_out_len]
            state.mean_sil_emb, state.n_sil_frames = self._get_silence_profile(
                state.mean_sil_emb, state.n_sil_frames, pop_out_embs, pop_out_preds
            )
            state.fifo = state.fifo[:, pop_out_len:]
            state.fifo_preds = state.fifo_preds[:, pop_out_len:]

            state.spkcache = torch.cat([state.spkcache, pop_out_embs], dim=1)
            if state.spkcache_preds is not None:
                state.spkcache_preds = torch.cat([state.spkcache_preds, pop_out_preds], dim=1)
            if state.spkcache.shape[1] > self.config.spkcache_len:
                if state.spkcache_preds is None:
                    state.spkcache_preds = torch.cat([preds[:, :spkcache_len], pop_out_preds], dim=1)
                state.spkcache, state.spkcache_preds = self._compress_spkcache(
                    state.spkcache, state.spkcache_preds, state.mean_sil_emb
                )

        return state, chunk_preds

    @torch.no_grad()
    def streaming_step(
        self,
        chunk_features: torch.Tensor,
        state: SortformerStreamingState,
        chunk_lengths: torch.Tensor | None = None,
        left_offset: int = 0,
        right_offset: int = 0,
        causal: bool = False,
        right_context: int = 7,
    ):
        """
        Process a single (already context-windowed) chunk of input features and update the streaming state.

        Args:
            chunk_features (`torch.FloatTensor` of shape `(batch_size, window_frames, num_mel_bins)`):
                Log-mel features for one chunk, including `left_offset` left-context and `right_offset` right-context
                input frames.
            state ([`SortformerStreamingState`]):
                The streaming state to read from and update (in place).
            chunk_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Valid feature-frame lengths within `chunk_features`. Defaults to the full window.
            left_offset (`int`, *optional*, defaults to 0):
                Number of left-context **input** feature frames included at the start of `chunk_features`.
            right_offset (`int`, *optional*, defaults to 0):
                Number of right-context **input** feature frames included at the end of `chunk_features`.

        Returns:
            `tuple(torch.FloatTensor, SortformerStreamingState)`: the chunk's per-frame speaker probabilities of shape
            `(batch_size, chunk_out_frames, num_speakers)` and the updated streaming state.
        """
        encoder = self.sortformer.encoder
        batch_size, window_frames, _ = chunk_features.shape
        if chunk_lengths is None:
            chunk_lengths = torch.full((batch_size,), window_frames, dtype=torch.long, device=chunk_features.device)

        chunk_mask = self._length_to_mask(chunk_lengths, window_frames)
        chunk_embs = encoder.subsampling(chunk_features, chunk_mask)
        chunk_emb_lengths = encoder._get_subsampling_output_length(chunk_lengths)

        # Concatenate [spkcache, fifo, chunk] subsampled embeddings (synchronous: batch-uniform lengths).
        concat_embs = torch.cat([state.spkcache, state.fifo, chunk_embs], dim=1)
        concat_lengths = state.spkcache.shape[1] + state.fifo.shape[1] + chunk_emb_lengths

        encoder_embs, _ = self._run_conformer_on_embeddings(
            concat_embs, concat_lengths, causal=causal, right_context=right_context
        )
        preds = self._forward_infer(encoder_embs, concat_lengths)
        preds = self._apply_mask_to_preds(preds, concat_lengths)

        lc = round(left_offset / self._subsampling_factor)
        rc = math.ceil(right_offset / self._subsampling_factor)
        state, chunk_preds = self._streaming_update(state, chunk=chunk_embs, preds=preds, lc=lc, rc=rc)
        return chunk_preds, state

    def _streaming_feat_loader(
        self, feat_seq: torch.Tensor, feat_seq_length: torch.Tensor
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, int]]:
        """Yield successive context-windowed chunks `(chunk_features, chunk_lengths, left_offset, right_offset)`."""
        subsampling_factor = self._subsampling_factor
        chunk_len = self.config.chunk_len
        left_context = self.config.chunk_left_context
        right_context = self.config.chunk_right_context
        feat_len = feat_seq.shape[2]
        feat_seq_offset = torch.zeros_like(feat_seq_length)

        start = 0
        end = 0
        while end < feat_len:
            left_offset = min(left_context * subsampling_factor, start)
            end = min(start + chunk_len * subsampling_factor, feat_len)
            right_offset = min(right_context * subsampling_factor, feat_len - end)
            chunk_feat_seq = feat_seq[:, :, start - left_offset : end + right_offset]
            feat_lengths = (feat_seq_length + feat_seq_offset - start + left_offset).clamp(0, chunk_feat_seq.shape[2])
            feat_lengths = feat_lengths * (feat_seq_offset < end)
            start = end
            yield torch.transpose(chunk_feat_seq, 1, 2), feat_lengths, left_offset, right_offset

    @torch.no_grad()
    def diarize_streaming(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
        right_context: int = 7,
    ) -> torch.Tensor:
        """
        Run full chunked streaming diarization over a feature sequence and return the concatenated per-frame speaker
        probabilities. This is the convenience wrapper around [`~SortformerForAudioFrameClassification.streaming_step`]
        and reproduces NeMo's synchronous `forward_streaming`.

        Unlike the offline [`forward`], the streaming path expects features extracted **without** waveform
        peak-normalization.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`):
                Log-mel filterbank features for the full audio, as produced by [`SortformerFeatureExtractor`].
            attention_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
                Mask marking valid input feature frames. Defaults to all frames being valid.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_frames, num_speakers)`: per-frame speaker-activity
            probabilities (multi-label sigmoid outputs).
        """
        feat_seq = input_features.transpose(1, 2)  # (batch, num_mel_bins, num_frames)
        batch_size, _, feat_len = feat_seq.shape
        if attention_mask is not None:
            feat_seq_length = attention_mask.sum(-1).to(torch.long)
        else:
            feat_seq_length = torch.full((batch_size,), feat_len, dtype=torch.long, device=feat_seq.device)

        state = self.init_streaming_state(batch_size=batch_size, device=feat_seq.device, dtype=input_features.dtype)
        total_preds = torch.zeros(
            (batch_size, 0, self.config.num_speakers), device=feat_seq.device, dtype=feat_seq.dtype
        )

        for chunk_features, chunk_lengths, left_offset, right_offset in self._streaming_feat_loader(
            feat_seq, feat_seq_length
        ):
            chunk_preds, state = self.streaming_step(
                chunk_features,
                state,
                chunk_lengths=chunk_lengths,
                left_offset=left_offset,
                right_offset=right_offset,
                causal=causal,
                right_context=right_context,
            )
            total_preds = torch.cat([total_preds, chunk_preds], dim=1)

        return total_preds


__all__ = [
    "SortformerModel",
    "SortformerForAudioFrameClassification",
    "SortformerPreTrainedModel",
    "SortformerStreamingState",
]
