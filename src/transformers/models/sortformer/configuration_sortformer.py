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
"""Sortformer model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig


@auto_docstring(checkpoint="nvidia/diar_streaming_sortformer_4spk-v2")
@strict
class SortformerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SortformerForAudioFrameClassification`]. It is
    used to instantiate a Sortformer end-to-end speaker-diarization model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a configuration similar
    to that of the NVIDIA NeMo Sortformer model
    [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2).

    Sortformer stacks a FastConformer acoustic encoder (shared with [`ParakeetEncoder`]) with a post-LN Transformer
    encoder and a per-frame sigmoid speaker-activity head. The model emits, for every output frame, an independent
    probability for each of `num_speakers` speakers being active (multi-label, hence the sigmoid).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        encoder_config (`dict` or `ParakeetEncoderConfig`, *optional*):
            Configuration of the FastConformer acoustic encoder. If `None`, a default `ParakeetEncoderConfig`
            matching the NeMo Sortformer FastConformer (NEST) encoder is used.
        hidden_size (`int`, *optional*, defaults to 192):
            Dimensionality of the Transformer encoder layers and the diarization head (NeMo `tf_d_model`). The
            FastConformer encoder output is linearly projected to this size when it differs from
            `encoder_config.hidden_size`.
        num_hidden_layers (`int`, *optional*, defaults to 18):
            Number of layers in the post-LN Transformer encoder that sits on top of the FastConformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 768):
            Dimensionality of the feed-forward (inner) layer of the Transformer encoder (NeMo `inner_size`).
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the Transformer encoder and the speaker head.
        hidden_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability for the feed-forward layers of the Transformer encoder (NeMo `ffn_dropout`).
        attention_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability applied to attention scores (NeMo `attn_score_dropout`).
        attention_layer_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability applied to the attention sublayer output (NeMo `attn_layer_dropout`).
        head_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability used inside the speaker-activity head (NeMo `dropout_rate`).
        pre_ln (`bool`, *optional*, defaults to `False`):
            Whether the Transformer encoder uses pre-layer-normalization. Sortformer uses post-LN (`False`).
        pre_ln_final_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply a final layer normalization at the output of the Transformer encoder stack.
        num_speakers (`int`, *optional*, defaults to 4):
            Maximum number of speakers the model can predict (NeMo `max_num_of_spks`). This is the number of
            independent sigmoid outputs per frame.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers of the Transformer encoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated-normal initializer for the linear/embedding layers.

    Example:

    ```python
    >>> from transformers import SortformerForAudioFrameClassification, SortformerConfig

    >>> configuration = SortformerConfig()
    >>> model = SortformerForAudioFrameClassification(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "sortformer"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    # FastConformer (NEST) encoder defaults matching
    # examples/speaker_tasks/diarization/conf/neural_diarizer/streaming_sortformer_diarizer_4spk-v2.yaml
    _default_encoder_config_kwargs = {
        "hidden_size": 512,  # model_defaults.fc_d_model
        "num_hidden_layers": 17,  # encoder.n_layers
        "num_attention_heads": 8,  # encoder.n_heads
        "intermediate_size": 2048,  # d_model * ff_expansion_factor (512 * 4)
        "hidden_act": "silu",  # Conformer uses Swish/SiLU
        "conv_kernel_size": 9,  # encoder.conv_kernel_size
        "subsampling_factor": 8,  # encoder.subsampling_factor
        "subsampling_conv_channels": 256,  # encoder.subsampling_conv_channels
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "num_mel_bins": 128,  # preprocessor.features
        "scale_input": True,  # encoder.xscaling
        "max_position_embeddings": 5000,  # encoder.pos_emb_max_len
        "dropout": 0.1,
        "dropout_positions": 0.0,
        "layerdrop": 0.0,
        "activation_dropout": 0.1,
        "attention_dropout": 0.1,
        "initializer_range": 0.02,
    }

    encoder_config: "dict | PreTrainedConfig | None" = None
    hidden_size: int = 192
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    intermediate_size: int = 768
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.5
    attention_dropout: float | int = 0.5
    attention_layer_dropout: float | int = 0.5
    head_dropout: float | int = 0.5
    pre_ln: bool = False
    pre_ln_final_layer_norm: bool = True
    num_speakers: int = 4
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "parakeet_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](
                **{**self._default_encoder_config_kwargs, **self.encoder_config}
            )
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["parakeet_encoder"](**self._default_encoder_config_kwargs)

        super().__post_init__(**kwargs)


__all__ = ["SortformerConfig"]
