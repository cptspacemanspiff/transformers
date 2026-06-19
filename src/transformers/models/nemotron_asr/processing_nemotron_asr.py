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

import torch

from ...audio_utils import AudioInput
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ..parakeet.processing_parakeet import ParakeetProcessor, ParakeetProcessorKwargs


logger = logging.get_logger(__name__)


class NemotronAsrProcessorKwargs(ParakeetProcessorKwargs, total=False):
    pass


@auto_docstring
class NemotronAsrProcessor(ParakeetProcessor):
    r"""
    Constructs a Nemotron 3.5 ASR processor wrapping a [`NemotronAsrFeatureExtractor`] and a [`NemotronAsrTokenizer`].

    In addition to [`ParakeetProcessor`], it knows the model's language-ID `prompt_dictionary` and can turn a
    `target_lang` (e.g. `"en-US"`, or `"auto"` for language detection) into the `prompt_indices` tensor that
    conditions the model.
    """

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        prompt_dictionary=None,
        blank_token="<blank>",
        decoder_type="rnnt",
    ):
        r"""
        prompt_dictionary (`dict[str, int]`, *optional*):
            Mapping from language-locale string (e.g. `"en-US"`, `"auto"`) to its prompt index.
        blank_token (`str`, *optional*, defaults to `"<blank>"`):
            Blank token for transducer decoding.
        decoder_type (`str`, *optional*, defaults to `"rnnt"`):
            Decoding/timestamp emission mode (always `"rnnt"` for this model).
        """
        self.prompt_dictionary = prompt_dictionary or {}
        super().__init__(feature_extractor, tokenizer, blank_token=blank_token, decoder_type=decoder_type)

    def prompt_indices_for(self, target_lang) -> torch.Tensor:
        """Map a language-locale string (or list of them) to a `prompt_indices` LongTensor."""
        langs = [target_lang] if isinstance(target_lang, str) else list(target_lang)
        idxs = []
        for lang in langs:
            if lang not in self.prompt_dictionary:
                available = list(self.prompt_dictionary)
                raise ValueError(
                    f"Unknown target language '{lang}'. Available: {available[:20]}"
                    f"{'...' if len(available) > 20 else ''}"
                )
            idxs.append(self.prompt_dictionary[lang])
        return torch.tensor(idxs, dtype=torch.long)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        target_lang: str | list[str] | None = None,
        sampling_rate: int | None = None,
        **kwargs: Unpack[NemotronAsrProcessorKwargs],
    ):
        r"""
        target_lang (`str` or `list[str]`, *optional*):
            Language-locale to condition transcription on (e.g. `"en-US"`, `"de-DE"`, or `"auto"`). Adds a
            `prompt_indices` entry to the returned inputs.
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. Validated against the feature extractor's expected rate.
        """
        inputs = super().__call__(audio, text=text, sampling_rate=sampling_rate, **kwargs)
        if target_lang is not None:
            prompt_indices = self.prompt_indices_for(target_lang)
            n = inputs["input_features"].shape[0]
            if prompt_indices.shape[0] == 1 and n > 1:
                prompt_indices = prompt_indices.expand(n)
            inputs["prompt_indices"] = prompt_indices
        return inputs


__all__ = ["NemotronAsrProcessor"]
