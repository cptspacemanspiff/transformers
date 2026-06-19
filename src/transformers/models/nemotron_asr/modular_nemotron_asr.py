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
"""PyTorch Nemotron 3.5 ASR model.

Nemotron 3.5 ASR (`nvidia/nemotron-3.5-asr-streaming-0.6b`) is a multilingual, prompt-conditioned cache-aware
streaming RNN-T. Its encoder is a *streaming* FastConformer (same family as Parakeet) with two differences: the
convolutions are causal and the self-attention uses a limited, chunked context window. A language-ID one-hot prompt
is fused with the encoder output (concat + MLP projection) before the RNN-T joint network.
"""

import math

import torch
from torch import nn

from ...activations import ACT2FN
from ...generation import GenerationMode
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..parakeet.modeling_parakeet import (
    ParakeetEncoderAttention,
    ParakeetEncoderBlock,
    ParakeetEncoderFeedForward,
    ParakeetEncoderModelOutput,
    ParakeetEncoderRelPositionalEncoding,
    ParakeetForRNNT,
    ParakeetPreTrainedModel,
    ParakeetRNNTDecoder,
    ParakeetRNNTJointNetwork,
)
from .configuration_nemotron_asr import NemotronAsrConfig, NemotronAsrEncoderConfig
from .generation_nemotron_asr import (
    NemotronAsrRNNTDecoderCache,
    NemotronAsrRNNTGenerationMixin,
    NemotronAsrStreamingCache,
)


logger = logging.get_logger(__name__)


class NemotronAsrEncoderModelOutput(ParakeetEncoderModelOutput):
    pass


class NemotronAsrEncoderRelPositionalEncoding(ParakeetEncoderRelPositionalEncoding):
    pass


class NemotronAsrEncoderFeedForward(ParakeetEncoderFeedForward):
    pass


class NemotronAsrEncoderAttention(ParakeetEncoderAttention):
    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Relative-position attention with a key/value left-context cache (one streaming chunk).

        `hidden_states` are the `n` new (query) frames; `key_value_states` are `[cache, new]` of length `T2`.
        """
        batch_size, query_length = hidden_states.shape[:2]
        kv_length = key_value_states.shape[1]
        num_heads = self.config.num_attention_heads

        query_states = self.q_proj(hidden_states).view(batch_size, query_length, num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = (
            self.k_proj(key_value_states).view(batch_size, kv_length, num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(key_value_states).view(batch_size, kv_length, num_heads, self.head_dim).transpose(1, 2)
        )

        query_with_bias_u = query_states + self.bias_u.view(1, num_heads, 1, self.head_dim)
        query_with_bias_v = query_states + self.bias_v.view(1, num_heads, 1, self.head_dim)

        relative_key_states = self.relative_k_proj(position_embeddings).view(batch_size, -1, num_heads, self.head_dim)
        matrix_bd = query_with_bias_v @ relative_key_states.permute(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)[..., :kv_length]
        matrix_ac = query_with_bias_u @ key_states.transpose(-2, -1)

        scores = (matrix_ac + matrix_bd) * self.scaling
        scores = scores.masked_fill(attention_mask[:, None].logical_not(), torch.finfo(scores.dtype).min)
        attn_weights = scores.softmax(dim=-1)
        attn_output = (attn_weights @ value_states).transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.o_proj(attn_output)


class NemotronAsrEncoderConvolutionModule(nn.Module):
    """Conformer convolution module for the streaming encoder.

    Differs from [`ParakeetEncoderConvolutionModule`] in two ways: the depthwise convolution is *causal* (left-padded
    by `kernel_size - 1`, no right padding) and the normalization is a `LayerNorm` over channels rather than a
    `BatchNorm1d`.
    """

    def __init__(self, config: NemotronAsrEncoderConfig):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size
        self.activation = ACT2FN[getattr(config, "hidden_act", "silu")]
        # causal depthwise conv: pad only on the left so each frame sees no future frames
        self.left_padding = kernel_size - 1

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=config.convolution_bias)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=0, groups=channels, bias=config.convolution_bias
        )
        if config.conv_norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=config.convolution_bias)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states.transpose(1, 2)  # (B, C, T)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # zero out padded time steps before the (causal) depthwise conv so they don't leak into valid frames
        if attention_mask is not None:
            pad_keep = attention_mask
            if pad_keep.dim() == 4:  # (B, 1, T, T) -> (B, T) keep mask along the key axis
                pad_keep = pad_keep.any(dim=1).any(dim=1)
            hidden_states = hidden_states.masked_fill(~pad_keep[:, None, :], 0.0)

        hidden_states = nn.functional.pad(hidden_states, (self.left_padding, 0))
        hidden_states = self.depthwise_conv(hidden_states)

        if isinstance(self.norm, nn.LayerNorm):
            hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        else:
            hidden_states = self.norm(hidden_states)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)

    def forward_streaming(self, hidden_states: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal conv module for one streaming chunk, using `cache` (the previous `kernel - 1` input frames)."""
        hidden_states = hidden_states.transpose(1, 2)  # (B, C, n)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        combined = torch.cat([cache, hidden_states], dim=-1)  # prepend cached left context instead of zero-padding
        new_cache = combined[:, :, -self.left_padding :]
        hidden_states = self.depthwise_conv(combined)

        if isinstance(self.norm, nn.LayerNorm):
            hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        else:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2), new_cache


class NemotronAsrEncoderSubsamplingConv2D(nn.Module):
    """Causal depthwise-striding 2D convolutional subsampling (8x time reduction).

    Identical structure to [`ParakeetEncoderSubsamplingConv2D`] but with *causal* padding on both the time and
    frequency axes (`pad = (kernel - 1, stride - 1)`), matching NeMo's `causal_downsampling=True`. This changes the
    surviving frequency dimension (e.g. 128 mel bins -> 17, not 16), hence the linear projection's input width.
    """

    def __init__(self, config: NemotronAsrEncoderConfig):
        super().__init__()
        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.causal = config.causal_downsampling
        self.num_layers = int(math.log2(config.subsampling_factor))

        if self.causal:
            self.left_padding = self.kernel_size - 1
            self.right_padding = self.stride - 1
        else:
            self.left_padding = self.right_padding = (self.kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, stride=self.stride, padding=0))
        self.layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1):
            self.layers.append(
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=0,
                    groups=self.channels,
                )
            )
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=1))
            self.layers.append(nn.ReLU())

        freq = config.num_mel_bins
        for _ in range(self.num_layers):
            freq = self._conv_out_length(freq)
        self.linear = nn.Linear(self.channels * freq, config.hidden_size, bias=True)

    def _conv_out_length(self, length):
        return (length + self.left_padding + self.right_padding - self.kernel_size) // self.stride + 1

    @staticmethod
    def _mask_time(hidden_states: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        # zero padded time steps (across channels and frequency) so they cannot leak into valid frames
        if lengths is None:
            return hidden_states
        time = hidden_states.shape[2]
        keep = torch.arange(time, device=hidden_states.device)[None, :] < lengths[:, None]
        return hidden_states * keep[:, None, :, None].to(hidden_states.dtype)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None):
        hidden_states = input_features.unsqueeze(1)  # (B, 1, T, F)
        # running valid time lengths, recomputed after every strided conv (mirrors NeMo's MaskedConvSequential)
        lengths = attention_mask.sum(-1).to(torch.long) if attention_mask is not None else None

        for layer in self.layers:
            # mask padded time steps before each layer: conv biases turn padded positions non-zero, so re-mask
            hidden_states = self._mask_time(hidden_states, lengths)
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1:
                hidden_states = nn.functional.pad(
                    hidden_states,
                    (self.left_padding, self.right_padding, self.left_padding, self.right_padding),
                )
            hidden_states = layer(hidden_states)
            if isinstance(layer, nn.Conv2d) and layer.stride[0] > 1 and lengths is not None:
                lengths = self._conv_out_length(lengths)
        hidden_states = self._mask_time(hidden_states, lengths)

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class NemotronAsrEncoderBlock(ParakeetEncoderBlock):
    def __init__(self, config: NemotronAsrEncoderConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.self_attn = NemotronAsrEncoderAttention(config, layer_idx)
        self.conv = NemotronAsrEncoderConvolutionModule(config)


@auto_docstring
class NemotronAsrPreTrainedModel(ParakeetPreTrainedModel):
    config: NemotronAsrConfig
    _no_split_modules = ["NemotronAsrEncoderBlock"]
    _can_record_outputs = {
        "hidden_states": NemotronAsrEncoderBlock,
        "attentions": NemotronAsrEncoderAttention,
    }

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)
        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))

        if encoder_config.causal_downsampling:
            all_paddings = (kernel_size - 1) + (stride - 1)
        else:
            all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size

        lengths = input_lengths
        for _ in range(num_layers):
            lengths = torch.floor(torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0)
        return lengths.to(dtype=torch.int)


def build_chunked_limited_mask(
    att_context_size, seq_length: int, device, style: str = "chunked_limited"
) -> torch.Tensor:
    """Boolean *keep* mask of shape `(seq_length, seq_length)` where `True` means query *i* attends to key *j*.

    Mirrors NeMo's `ConformerEncoder._create_masks`. For `chunked_limited`, frames are grouped into
    non-overlapping chunks of size `right + 1`; chunk `c` attends to its own chunk and `left // (right + 1)` chunks
    to the left.
    """
    left, right = int(att_context_size[0]), int(att_context_size[1])
    idx = torch.arange(seq_length, device=device)

    if style == "chunked_limited" and right != -1:
        chunk_size = right + 1
        left_chunks = left // chunk_size if left >= 0 else 10000
        chunk_idx = torch.div(idx, chunk_size, rounding_mode="trunc")
        diff = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
        return torch.logical_and(diff >= 0, diff <= left_chunks)

    # regular sliding window (and unlimited-right fallback)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # i - j
    keep = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
    if left >= 0:
        keep = torch.logical_and(keep, diff <= left)
    if right >= 0:
        keep = torch.logical_and(keep, diff >= -right)
    return keep


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 ASR cache-aware streaming FastConformer encoder.
    """
)
class NemotronAsrEncoder(NemotronAsrPreTrainedModel):
    config: NemotronAsrEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config: NemotronAsrEncoderConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop

        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = NemotronAsrEncoderSubsamplingConv2D(config)
        self.encode_positions = NemotronAsrEncoderRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [NemotronAsrEncoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attention_mask: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NemotronAsrEncoderModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the subsampled attention mask. Only effective when `attention_mask` is provided.
        """
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
        )

        seq_length = hidden_states.shape[1]
        # limited, chunked context applies regardless of padding
        band_mask = build_chunked_limited_mask(
            self.config.att_context_size, seq_length, hidden_states.device, self.config.att_context_style
        )

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=seq_length)
            pad_mask = output_mask.unsqueeze(1).expand(-1, seq_length, -1)
            pad_mask = pad_mask & pad_mask.transpose(1, 2)
            attention_mask = (pad_mask & band_mask.unsqueeze(0)).unsqueeze(1)
        else:
            attention_mask = band_mask.view(1, 1, seq_length, seq_length)

        for encoder_layer in self.layers:
            to_drop = self.training and torch.rand([]) < self.layerdrop
            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        return NemotronAsrEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_mask is not None and output_attention_mask else None,
        )

    @property
    def streaming_config(self) -> dict:
        """Derived cache-aware streaming sizes for the configured `att_context_size` (all in 80 ms encoder frames,
        except mel sizes). Mirrors NeMo's `ConformerEncoder.setup_streaming_params` for the `chunked_limited` style.
        """
        config = self.config
        chunk = config.att_context_size[1] + 1
        subsampling_factor = config.subsampling_factor
        pre_encode_frames = subsampling_factor + 1
        return {
            "last_channel_cache_size": config.att_context_size[0],
            "conv_cache_size": config.conv_kernel_size - 1,
            "chunk_frames": chunk,  # subsampled output frames emitted per step
            "left_chunks": config.att_context_size[0] // chunk,
            "valid_out_len": chunk,
            "pre_encode_frames": pre_encode_frames,
            "drop_extra_pre_encoded": 1 + (pre_encode_frames - 1) // subsampling_factor,
            "chunk_mel": chunk * subsampling_factor,
            "first_chunk_mel": chunk * subsampling_factor - (subsampling_factor - 1),
        }

    def _build_streaming_attention_mask(self, query_length: int, valid_cache_len: int, device) -> torch.Tensor:
        """Chunked-limited keep-mask `(1, query_length, cache + query_length)` for one streaming chunk.

        The new (query) frames sit at the end of the `[cache, new]` sequence; unfilled cache slots are masked out.
        """
        scfg = self.streaming_config
        cache_size, chunk, left_chunks = scfg["last_channel_cache_size"], scfg["chunk_frames"], scfg["left_chunks"]
        kv_length = cache_size + query_length
        idx = torch.arange(kv_length, device=device)
        chunk_idx = torch.div(idx, chunk, rounding_mode="trunc")
        diff = chunk_idx[:, None] - chunk_idx[None, :]
        band = (diff >= 0) & (diff <= left_chunks)
        valid_keys = idx >= (cache_size - valid_cache_len)  # ignore not-yet-filled cache positions
        keep = band & valid_keys[None, :]
        return keep[cache_size:].unsqueeze(0)

    def streaming_forward(
        self,
        input_features: torch.Tensor,
        last_channel: list[torch.Tensor],
        last_time: list[torch.Tensor],
        last_channel_len: int,
        drop_extra_pre_encoded: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], int]:
        """Run the cache-aware encoder on one chunk of (already left-padded) mel features.

        Returns the chunk embeddings `(batch, valid_out_len, hidden)` and the updated per-layer caches. This is the
        numerical equivalent of NeMo's `ConformerEncoder.cache_aware_stream_step` for the `chunked_limited` style.
        """
        scfg = self.streaming_config
        cache_size = scfg["last_channel_cache_size"]

        hidden_states = self.subsampling(input_features)
        hidden_states = hidden_states[:, drop_extra_pre_encoded:]
        hidden_states = hidden_states * self.input_scale
        query_length = hidden_states.shape[1]

        position_embeddings = self.encode_positions(
            hidden_states.new_zeros(hidden_states.shape[0], query_length + cache_size, hidden_states.shape[2])
        )
        attention_mask = self._build_streaming_attention_mask(query_length, last_channel_len, hidden_states.device)

        new_channel, new_time = [], []
        for i, layer in enumerate(self.layers):
            hidden_states = hidden_states + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(hidden_states))
            normed = layer.norm_self_att(hidden_states)
            key_value_states = torch.cat([last_channel[i], normed], dim=1) if last_channel[i].shape[1] else normed
            attn_output = layer.self_attn.forward_streaming(
                normed, position_embeddings, key_value_states, attention_mask
            )
            new_channel.append(key_value_states[:, -cache_size:])
            hidden_states = hidden_states + attn_output

            conv_output, conv_cache = layer.conv.forward_streaming(layer.norm_conv(hidden_states), last_time[i])
            new_time.append(conv_cache)
            hidden_states = hidden_states + conv_output

            hidden_states = hidden_states + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(hidden_states))
            hidden_states = layer.norm_out(hidden_states)

        hidden_states = hidden_states[:, : scfg["valid_out_len"]]
        next_cache_len = min(last_channel_len + query_length, cache_size)
        return hidden_states, new_channel, new_time, next_cache_len


class NemotronAsrRNNTDecoder(ParakeetRNNTDecoder):
    def __init__(self, config: NemotronAsrConfig):
        super().__init__(config)


class NemotronAsrRNNTJointNetwork(ParakeetRNNTJointNetwork):
    def __init__(self, config: NemotronAsrConfig):
        super().__init__(config)


@auto_docstring(
    custom_intro="""
    Nemotron 3.5 ASR: a prompt-conditioned cache-aware streaming FastConformer encoder with an RNN-T head.

    A one-hot language prompt (selected by `prompt_indices`) is broadcast across time, concatenated to the encoder
    output, and projected back to `hidden_size` by `prompt_kernel` before the RNN-T joint network.
    """
)
class NemotronAsrForRNNT(NemotronAsrRNNTGenerationMixin, ParakeetForRNNT):
    config: NemotronAsrConfig
    _no_split_modules = ["NemotronAsrEncoderBlock", "NemotronAsrRNNTDecoder"]
    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH]

    def __init__(self, config: NemotronAsrConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.num_prompts = config.num_prompts
        # prompt_kernel: concat(encoder_out, one_hot_lang) -> MLP -> back to hidden_size
        self.prompt_kernel = nn.Sequential(
            nn.Linear(config.encoder_config.hidden_size + config.num_prompts, config.encoder_config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.encoder_config.hidden_size * 2, config.encoder_config.hidden_size),
        )
        self.encoder_projector = nn.Linear(config.encoder_config.hidden_size, config.decoder_hidden_size)
        self.decoder = NemotronAsrRNNTDecoder(config)
        self.joint = NemotronAsrRNNTJointNetwork(config)
        self.max_symbols_per_step = config.max_symbols_per_step
        self.post_init()

    def _apply_prompt(self, encoder_hidden_states: torch.Tensor, prompt_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = encoder_hidden_states.shape
        prompt = torch.zeros(
            batch_size,
            seq_length,
            self.num_prompts,
            dtype=encoder_hidden_states.dtype,
            device=encoder_hidden_states.device,
        )
        prompt.scatter_(2, prompt_indices.view(batch_size, 1, 1).expand(-1, seq_length, -1).long(), 1.0)
        fused = torch.cat([encoder_hidden_states, prompt], dim=-1)
        return self.prompt_kernel(fused)

    @can_return_tuple
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NemotronAsrEncoderModelOutput:
        encoder_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask, **kwargs)
        hidden_states = encoder_outputs.last_hidden_state
        if prompt_indices is not None:
            hidden_states = self._apply_prompt(hidden_states, prompt_indices)
        encoder_outputs.last_hidden_state = hidden_states
        encoder_outputs.pooler_output = self.encoder_projector(hidden_states)
        return encoder_outputs

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        prompt_indices: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_cache=None,
        use_decoder_cache: bool | None = None,
        encoder_outputs: NemotronAsrEncoderModelOutput | tuple[torch.FloatTensor] | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        prompt_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Language-ID prompt index per batch element (see `config.prompt_dictionary`). Selects the one-hot language
            vector fused with the encoder output. Required to reproduce the multilingual behavior of the model.
        decoder_cache (`NemotronAsrRNNTDecoderCache`, *optional*):
            Decoder LSTM cache. When provided and initialized, the cached `decoder_output` is reused (e.g. during
            blank-skipping) instead of running the decoder.
        use_decoder_cache (`bool`, *optional*):
            Whether to use a decoder cache. When `True` and `decoder_cache` is `None`, a new cache is created.
        """
        if encoder_outputs is None:
            encoder_outputs = self.get_audio_features(
                input_features=input_features,
                attention_mask=attention_mask,
                prompt_indices=prompt_indices,
                **kwargs,
            )
        return super().forward(
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_cache=decoder_cache,
            use_decoder_cache=use_decoder_cache,
            encoder_outputs=encoder_outputs,
            labels=labels,
            **kwargs,
        )

    # ----- cache-aware streaming inference -----

    def init_streaming_state(self, batch_size: int = 1, device=None, dtype=None) -> NemotronAsrStreamingCache:
        """Create an empty [`NemotronAsrStreamingCache`] for cache-aware streaming inference."""
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        encoder_config = self.config.encoder_config
        scfg = self.encoder.streaming_config
        hidden_size = encoder_config.hidden_size
        decoder_cache = NemotronAsrRNNTDecoderCache(self.config)
        return NemotronAsrStreamingCache(
            # full-size channel caches; the streaming mask ignores the not-yet-filled (zero) left part
            last_channel=[
                torch.zeros(batch_size, scfg["last_channel_cache_size"], hidden_size, device=device, dtype=dtype)
                for _ in range(encoder_config.num_hidden_layers)
            ],
            last_time=[
                torch.zeros(batch_size, hidden_size, scfg["conv_cache_size"], device=device, dtype=dtype)
                for _ in range(encoder_config.num_hidden_layers)
            ],
            last_channel_len=0,
            pre_encode=torch.zeros(batch_size, 0, encoder_config.num_mel_bins, device=device, dtype=dtype),
            decoder_cache=decoder_cache,
            last_token=torch.full((batch_size, 1), self.config.blank_token_id, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def _streaming_decode(self, encoder_frames: torch.Tensor, cache: NemotronAsrStreamingCache) -> list[list[int]]:
        """Greedy RNN-T decode of one chunk of (projected) encoder frames, carrying the decoder state in `cache`."""
        blank_token_id = self.config.blank_token_id
        batch_size, num_frames = encoder_frames.shape[0], encoder_frames.shape[1]
        if not cache.decoder_cache.is_initialized:
            self.decoder(cache.last_token, cache=cache.decoder_cache)  # g_0 for the start (blank) token

        emitted: list[list[int]] = [[] for _ in range(batch_size)]
        for t in range(num_frames):
            encoder_step = encoder_frames[:, t]  # (batch, decoder_hidden)
            for _ in range(self.max_symbols_per_step):
                logits = self.joint(
                    encoder_hidden_states=encoder_step[:, None, None, :],
                    decoder_hidden_states=cache.decoder_cache.cache[:, None, :, :],
                ).reshape(batch_size, -1)
                tokens = logits.argmax(dim=-1)
                if bool((tokens == blank_token_id).all()):
                    break
                for b in range(batch_size):
                    if int(tokens[b]) != blank_token_id:
                        emitted[b].append(int(tokens[b]))
                # advance the prediction network for non-blank emissions (blank elements are masked, state frozen)
                self.decoder(tokens[:, None], cache=cache.decoder_cache)
                cache.last_token = tokens[:, None]
        return emitted

    @torch.no_grad()
    def streaming_step(
        self,
        input_features: torch.Tensor,
        cache: NemotronAsrStreamingCache,
        prompt_indices: torch.Tensor | None = None,
        drop_extra_pre_encoded: int = 0,
    ) -> tuple[list[list[int]], NemotronAsrStreamingCache]:
        """Process one chunk of mel features (already left-padded with `cache.pre_encode`) and emit RNN-T tokens.

        Returns the per-batch list of newly emitted token ids and the updated `cache`.
        """
        embeddings, cache.last_channel, cache.last_time, cache.last_channel_len = self.encoder.streaming_forward(
            input_features, cache.last_channel, cache.last_time, cache.last_channel_len, drop_extra_pre_encoded
        )
        if prompt_indices is not None:
            embeddings = self._apply_prompt(embeddings, prompt_indices)
        encoder_frames = self.encoder_projector(embeddings)
        tokens = self._streaming_decode(encoder_frames, cache)
        return tokens, cache

    @torch.no_grad()
    def transcribe_stream(
        self,
        input_features: torch.Tensor,
        prompt_indices: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Convenience driver: split `input_features` into cache-aware streaming chunks and return emitted token ids.

        This reproduces NeMo's chunk schedule for the `chunked_limited` style (first chunk shorter, subsequent chunks
        left-padded by `pre_encode_frames`), so the output is identical to a real per-chunk stream.
        """
        scfg = self.encoder.streaming_config
        pre, chunk_mel, first_mel = scfg["pre_encode_frames"], scfg["chunk_mel"], scfg["first_chunk_mel"]
        drop = scfg["drop_extra_pre_encoded"]
        total = input_features.shape[1]

        cache = self.init_streaming_state(input_features.shape[0], input_features.device, input_features.dtype)
        emitted: list[list[int]] = [[] for _ in range(input_features.shape[0])]
        pos, step = 0, 0
        while pos < total:
            if step == 0:
                window, cur_drop = input_features[:, :first_mel], 0
                pos = first_mel
            else:
                end = min(pos + chunk_mel, total)
                window, cur_drop = input_features[:, pos - pre : end], drop
                pos = end
            tokens, cache = self.streaming_step(window, cache, prompt_indices, cur_drop)
            for b in range(len(emitted)):
                emitted[b].extend(tokens[b])
            step += 1
        return emitted


__all__ = ["NemotronAsrForRNNT", "NemotronAsrEncoder", "NemotronAsrPreTrainedModel"]
