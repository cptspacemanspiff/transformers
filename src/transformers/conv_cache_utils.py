# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Convolution caches for cache-aware streaming of (depthwise) 1D convolutions.

A PARALLEL hierarchy to the attention `Cache`/`CacheLayerMixin` design in `cache_utils.py`: a depthwise conv has no
notion of heads, mask sizes, or `kv_offset`, so subclassing the attention `Cache` (whose `get_mask_sizes` /
`get_max_cache_shape` are attention-specific) would be wrong. See the class docstrings below for details.
"""

from abc import ABC, abstractmethod

import torch

from .utils import is_torchdynamo_compiling


# ----------------------------------------------------------------------------------------------------------------------
# Convolution caches for cache-aware streaming of (depthwise) 1D convolutions.
#
# These mirror the attention `Cache`/`CacheLayerMixin` design, but as a PARALLEL hierarchy: a depthwise conv has no
# notion of heads, mask sizes, or `kv_offset`, so subclassing the attention `Cache` (whose `get_mask_sizes` /
# `get_max_cache_shape` are attention-specific) would be wrong. Instead, a `ConvCache` caches, per layer, the previous
# `kernel - 1` input frames (the causal left-context) of a 1D conv with `(B, C, T)` layout. Calling `update(frames,
# layer_idx)` prepends that context, returning a `(B, C, ctx + T)` tensor that is conv-ready for a `padding=0`
# convolution (so the output has the same length `T` as `frames` while remaining causal), and retains the new last
# `kernel - 1` frames for the next call. This generalizes the single-frame Mamba `update_conv_state` pattern (see
# `LinearAttentionLayer.update_conv_state`) to a chunk of frames, decoupled from any SSM/recurrent state.
# ----------------------------------------------------------------------------------------------------------------------


class ConvCacheLayerMixin(ABC):
    """Base, abstract class for a single layer's depthwise-convolution left-context cache.

    Stores the previous `kernel - 1` input frames of a causal 1D conv (`(B, C, kernel - 1)`), lazily initialized to
    zeros (the causal left-pad from `t = 0`). `update(hidden_states)` takes the new frames `(B, C, T)`, prepends the
    cached context, returns the `(B, C, (kernel - 1) + T)` conv-ready tensor, and retains the new trailing
    `kernel - 1` frames.
    """

    is_compileable = False

    def __init__(self):
        self.conv_states: torch.Tensor | None = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, hidden_states: torch.Tensor) -> None: ...

    @abstractmethod
    def update(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor: ...

    @abstractmethod
    def get_context_length(self) -> int: ...

    def reset(self) -> None:
        """Resets the cached context to zeros while preserving the object/buffers."""
        if self.is_initialized:
            self.conv_states.zero_()


class DynamicConvCacheLayer(ConvCacheLayerMixin):
    """A conv-cache layer that re-allocates (cat then keep-last-`kernel - 1`) on every update (eager / dev).

    Mirrors `DynamicLayer`. The cache tensor is replaced (not mutated in place) each step, so it is not
    `torch.compile` / cudagraph friendly, but is simple and shape-robust.

    Args:
        kernel_size (`int`): convolution kernel size; the cache holds `kernel_size - 1` left-context frames.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.context_length = kernel_size - 1

    def lazy_initialization(self, hidden_states: torch.Tensor) -> None:
        self.dtype, self.device = hidden_states.dtype, hidden_states.device
        batch_size, channels = hidden_states.shape[0], hidden_states.shape[1]
        self.conv_states = torch.zeros(
            (batch_size, channels, self.context_length), dtype=self.dtype, device=self.device
        )
        self.is_initialized = True

    def update(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(hidden_states)

        combined = torch.cat([self.conv_states, hidden_states], dim=-1)
        if self.context_length > 0:
            self.conv_states = combined[:, :, -self.context_length :]
        return combined

    def get_context_length(self) -> int:
        return self.context_length


class StaticConvCacheLayer(ConvCacheLayerMixin):
    """A conv-cache layer with a preallocated fixed `(B, C, kernel - 1)` buffer, mutated in place (export / compile).

    Mirrors `StaticLayer`: the backing buffer is allocated once (lazily) and updated via `index_copy_` / `roll`, so the
    data pointer and all shapes are static. This is the chunked generalization of `LinearAttentionLayer.update_conv_state`
    (Mamba), decoupled from the SSM/recurrent state.

    Args:
        kernel_size (`int`): convolution kernel size; the cache holds `kernel_size - 1` left-context frames.
    """

    is_compileable = True

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.context_length = kernel_size - 1

    def lazy_initialization(self, hidden_states: torch.Tensor) -> None:
        self.dtype, self.device = hidden_states.dtype, hidden_states.device
        self.max_batch_size, self.channels = hidden_states.shape[0], hidden_states.shape[1]
        self.conv_states = torch.zeros(
            (self.max_batch_size, self.channels, self.context_length), dtype=self.dtype, device=self.device
        )
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.conv_states)
        self.is_initialized = True

    def update(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(hidden_states)

        # Prepend the static left-context to the new frames; the returned tensor is conv-ready (padding=0).
        combined = torch.cat([self.conv_states, hidden_states], dim=-1)

        # Refresh the static buffer in place with the new trailing `kernel - 1` frames (keeps the data pointer).
        if self.context_length > 0:
            num_new_tokens = hidden_states.shape[-1]
            if num_new_tokens >= self.context_length:
                self.conv_states.copy_(hidden_states[:, :, -self.context_length :])
            else:
                rolled = self.conv_states.roll(shifts=-num_new_tokens, dims=-1)
                rolled[:, :, -num_new_tokens:] = hidden_states
                self.conv_states.copy_(rolled)

        return combined

    def get_context_length(self) -> int:
        return self.context_length


class ConvCache:
    """A container of per-layer `ConvCacheLayerMixin` objects — the convolution analog of `Cache`.

    Mirrors the attention `Cache` API (`update`, `reset`, lazy per-layer growth via `layer_class_to_replicate`) but for
    causal 1D depthwise convolutions: there are no heads, no mask sizes, and no `kv_offset`.

    Args:
        layers (`list[ConvCacheLayerMixin]`, *optional*): pre-created per-layer conv caches. If omitted,
            `layer_class_to_replicate` is used to lazily append layers as `update` is called.
        layer_class_to_replicate (`Callable[[], ConvCacheLayerMixin]`, *optional*): a zero-arg factory used to create
            each layer lazily (so the kernel size is already bound in the factory).
    """

    def __init__(
        self,
        layers: list["ConvCacheLayerMixin"] | None = None,
        layer_class_to_replicate=None,
    ):
        if (layers is None) == (layer_class_to_replicate is None):
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a ConvCache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def update(self, hidden_states: torch.Tensor, layer_idx: int, cache_kwargs: dict | None = None) -> torch.Tensor:
        """Prepend the layer's cached `kernel - 1` left-context to `hidden_states` `(B, C, T)` and return
        `(B, C, (kernel - 1) + T)`, retaining the new trailing context for the next call."""
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())
        return self.layers[layer_idx].update(hidden_states, **(cache_kwargs or {}))

    def get_context_length(self, layer_idx: int = 0) -> int:
        """Return the number of left-context frames (`kernel - 1`) the layer prepends per update."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_context_length()

    def reset(self) -> None:
        """Reset all layers' cached context to zeros."""
        for layer in self.layers:
            layer.reset()

    @property
    def is_compileable(self) -> bool:
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    def __len__(self):
        return len(self.layers)


class DynamicConvCache(ConvCache):
    """A `ConvCache` that grows lazily, one `DynamicConvCacheLayer` per layer (eager / dev). Like `DynamicCache`.

    Args:
        kernel_size (`int`): convolution kernel size shared by all layers (cache holds `kernel_size - 1` frames).
        num_layers (`int`, *optional*): if given, eagerly create this many layers; otherwise layers are appended
            lazily on first `update` per `layer_idx`.
    """

    def __init__(self, kernel_size: int, num_layers: int | None = None):
        self.kernel_size = kernel_size
        if num_layers is not None:
            super().__init__(layers=[DynamicConvCacheLayer(kernel_size) for _ in range(num_layers)])
        else:
            super().__init__(layer_class_to_replicate=lambda: DynamicConvCacheLayer(kernel_size))


class StaticConvCache(ConvCache):
    """A `ConvCache` with preallocated fixed-shape per-layer buffers (export / `torch.compile`). Like `StaticCache`.

    All layers are created eagerly and (optionally) initialized ahead-of-time so every `update` produces fixed-shape
    buffers with no dynamic-shape allocation.

    Args:
        kernel_size (`int`): convolution kernel size (cache holds `kernel_size - 1` frames per layer).
        num_layers (`int`): number of conv layers.
        channels (`int`): per-layer channel count `C`.
        batch_size (`int`, *optional*, defaults to 1): batch size `B` (used only for ahead-of-time initialization).
        device (`torch.device`, *optional*): device for ahead-of-time initialization.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`): dtype for ahead-of-time initialization.
        initialize (`bool`, *optional*, defaults to `True`): if `True`, allocate the static buffers immediately (so
            the shapes/data-pointers are fixed before the first `update`, as required by `torch.export`).
    """

    is_compileable = True

    def __init__(
        self,
        kernel_size: int,
        num_layers: int,
        channels: int,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        initialize: bool = True,
    ):
        self.kernel_size = kernel_size
        layers = [StaticConvCacheLayer(kernel_size) for _ in range(num_layers)]
        super().__init__(layers=layers)
        if initialize:
            fake = torch.zeros((batch_size, channels, 0), dtype=dtype, device=device)
            for layer in self.layers:
                layer.lazy_initialization(fake)
