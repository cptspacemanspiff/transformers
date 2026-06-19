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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/nemotron-3.5-asr-streaming-0.6b")
@strict
class NemotronAsrEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NemotronAsrEncoder`]. It is the cache-aware
    streaming FastConformer encoder of [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

    It differs from [`ParakeetEncoderConfig`] in that it is a *cache-aware streaming* encoder: convolutions are causal
    (`causal_downsampling`, causal depthwise conv) and self-attention uses a limited, chunked context window
    (`att_context_size`, `att_context_style`) rather than full bidirectional attention.

    convolution_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in convolutions of the conformer's convolution module.
    conv_kernel_size (`int`, *optional*, defaults to 9):
        The kernel size of the convolution layers in the Conformer block.
    conv_norm_type (`str`, *optional*, defaults to `"layer_norm"`):
        Normalization applied inside the conformer convolution module (`"layer_norm"` or `"batch_norm"`).
    causal_downsampling (`bool`, *optional*, defaults to `True`):
        Whether the subsampling convolutions are causal (left-padded), as used by the streaming model.
    subsampling_factor (`int`, *optional*, defaults to 8):
        The factor by which the input sequence is subsampled.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        The number of channels in the subsampling convolution layers.
    num_mel_bins (`int`, *optional*, defaults to 128):
        Number of mel features.
    subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the subsampling convolution layers.
    subsampling_conv_stride (`int`, *optional*, defaults to 2):
        The stride of the subsampling convolution layers.
    att_context_size (`list[int]`, *optional*, defaults to `[56, 13]`):
        The `[left, right]` attention context, in number of (80 ms) encoder frames, used to build the limited-context
        attention mask. `left`/`right` of `-1` mean unlimited context on that side.
    att_context_style (`str`, *optional*, defaults to `"chunked_limited"`):
        How the limited-context attention mask is built (`"chunked_limited"` or `"regular"`).
    scale_input (`bool`, *optional*, defaults to `False`):
        Whether to scale the subsampled embeddings by `sqrt(hidden_size)`.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the encoder layers.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of conformer encoder layers.
    num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer.
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the feed-forward layers.
    hidden_act (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function in the encoder.
    attention_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the query/key/value/output projections of self-attention.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the encoder.
    dropout_positions (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the positional embeddings.
    layerdrop (`float`, *optional*, defaults to 0.1):
        The LayerDrop probability for the encoder layers.
    activation_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for activations inside the feed-forward layers.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 5000):
        The maximum sequence length for the relative positional encoding.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated normal initializer.

    Example:
    ```python
    >>> from transformers import NemotronAsrEncoder, NemotronAsrEncoderConfig

    >>> configuration = NemotronAsrEncoderConfig()
    >>> model = NemotronAsrEncoder(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = False
    convolution_bias: bool = False
    conv_kernel_size: int = 9
    conv_norm_type: str = "layer_norm"
    causal_downsampling: bool = True
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 128
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    att_context_size: list[int] | tuple[int, int] = (56, 13)
    att_context_style: str = "chunked_limited"
    dropout: float | int = 0.1
    dropout_positions: float | int = 0.0
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    max_position_embeddings: int = 5000
    scale_input: bool = False
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/nemotron-3.5-asr-streaming-0.6b")
@strict
class NemotronAsrConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NemotronAsrForRNNT`], the multilingual,
    prompt-conditioned cache-aware streaming RNN-T model
    [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

    It extends the RNN-T configuration with the language-ID *prompt* mechanism: a one-hot language vector of width
    `num_prompts` is broadcast across time, concatenated to the encoder output, and projected back to `hidden_size`
    by a small MLP (`prompt_kernel`) before the RNN-T joint network.

    vocab_size (`int`, *optional*, defaults to 13088):
        Vocabulary size of the joint network (including the blank token).
    decoder_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the LSTM prediction network and joint network.
    num_decoder_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers in the prediction network.
    hidden_act (`str`, *optional*, defaults to `"relu"`):
        Activation used by the joint network.
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum number of symbols to emit per encoder time step during greedy decoding.
    num_prompts (`int`, *optional*, defaults to 128):
        Width of the one-hot language prompt vector concatenated to the encoder output.
    prompt_dictionary (`dict[str, int]`, *optional*):
        Mapping from language-locale string (e.g. `"en-US"`, `"auto"`) to its prompt index in `[0, num_prompts)`.
    encoder_config (`Union[dict, NemotronAsrEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.
    pad_token_id (`int`, *optional*, defaults to 13087):
        Padding token id. For RNN-T, this is the blank token reused as pad.
    blank_token_id (`int`, *optional*, defaults to 13087):
        Blank token id.

    Example:
    ```python
    >>> from transformers import NemotronAsrForRNNT, NemotronAsrConfig

    >>> configuration = NemotronAsrConfig()
    >>> model = NemotronAsrForRNNT(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr"
    sub_configs = {"encoder_config": NemotronAsrEncoderConfig}

    vocab_size: int = 13088
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    num_prompts: int = 128
    prompt_dictionary: dict | None = None
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 13087
    blank_token_id: int = 13087
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = NemotronAsrEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = NemotronAsrEncoderConfig()
        if self.prompt_dictionary is None:
            self.prompt_dictionary = {}
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


__all__ = ["NemotronAsrConfig", "NemotronAsrEncoderConfig"]
