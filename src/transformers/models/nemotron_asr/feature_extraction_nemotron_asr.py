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

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
from ..parakeet.feature_extraction_parakeet import EPSILON, ParakeetFeatureExtractor


logger = logging.get_logger(__name__)


class NemotronAsrFeatureExtractor(ParakeetFeatureExtractor):
    r"""
    Constructs a Nemotron 3.5 ASR feature extractor. It extracts log-mel-filter-bank features identically to
    [`ParakeetFeatureExtractor`], with two differences matching NeMo's `AudioToMelSpectrogramPreprocessor` for this
    model: it produces `feature_size=128` mel bins and it does **not** apply per-feature mean/variance normalization
    (`normalize="NA"`).

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        n_fft (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        win_length (`int`, *optional*, defaults to 400):
            The window length for the STFT computation.
        preemphasis (`float`, *optional*, defaults to 0.97):
            A preemphasis filter coefficient. 0.0 means no preemphasis filter.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to apply per-feature zero-mean unit-variance normalization to the mel features. Nemotron 3.5 ASR
            is trained with `normalize: "NA"`, so this defaults to `False` (matching [`SortformerFeatureExtractor`]).
    """

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        do_normalize=False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            preemphasis=preemphasis,
            padding_value=padding_value,
            **kwargs,
        )
        self.do_normalize = do_normalize

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor was trained using a sampling rate of "
                f"{self.sampling_rate}. Please make sure that the provided `raw_speech` was sampled with "
                f"{self.sampling_rate} and not {sampling_rate}."
            )

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_torch or is_batched_sequence:
            raw_speech = [speech[:, None].to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech[:, None].to(torch.float32)]

        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_features = padded_inputs.input_features.squeeze(-1)

        if self.preemphasis is not None:
            timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
                0
            ) < padded_inputs.audio_lengths.unsqueeze(1)
            input_features = torch.cat(
                [input_features[:, :1], input_features[:, 1:] - self.preemphasis * input_features[:, :-1]], dim=1
            )
            input_features = input_features.masked_fill(~timemask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device)
        features_lengths = torch.floor_divide(
            padded_inputs.audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
        )
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        mask = attention_mask.unsqueeze(-1)
        if do_normalize:
            input_features_masked = input_features * mask
            mean = (input_features_masked.sum(dim=1) / features_lengths.unsqueeze(-1)).unsqueeze(1)
            variance = ((input_features_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
            std = torch.sqrt(variance).unsqueeze(1)
            input_features = (input_features - mean) / (std + EPSILON)
        input_features = input_features * mask

        return BatchFeature(
            data={"input_features": input_features, "attention_mask": attention_mask},
            tensor_type=return_tensors,
        )


__all__ = ["NemotronAsrFeatureExtractor"]
