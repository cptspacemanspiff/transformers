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
"""Feature extractor class for Sortformer."""

import numpy as np
import torch

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_librosa_available, logging
from ...utils.import_utils import requires


if is_librosa_available():
    import librosa


EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24


logger = logging.get_logger(__name__)


@requires(backends=("torch", "librosa"))
class SortformerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Sortformer feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    It extracts 128-dimensional log-mel filterbank features matching NeMo's `AudioToMelSpectrogramPreprocessor`
    (`normalize: "NA"`) used by the Sortformer diarization model — i.e. it does **not** apply per-feature mean/variance
    normalization by default, unlike the Parakeet/CohereAsr extractors.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients (NeMo
            `window_stride` of 0.01s at 16kHz).
        n_fft (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        win_length (`int`, *optional*, defaults to 400):
            The window length for the STFT computation (NeMo `window_size` of 0.025s at 16kHz).
        preemphasis (`float`, *optional*, defaults to 0.97):
            A preemphasis filter coefficient. 0.0 means no preemphasis filter.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to apply per-feature zero-mean unit-variance normalization to the mel features. Sortformer is
            trained with `normalize: "NA"`, so this defaults to `False`.
    """

    model_input_names = ["input_features", "attention_mask"]

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
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.do_normalize = do_normalize

        # NeMo's FilterbankFeatures builds mel filters via librosa (slaney norm). We mirror that exactly so the
        # converted weights match. Using librosa keeps float32/float64 behaviour identical to the original.
        mel_filters = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=feature_size, fmin=0.0, fmax=sampling_rate / 2, norm="slaney"
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        # spectrogram
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        # log mel spectrogram
        mel_filters = self.mel_filters.to(device)
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)

        # (batch_size, num_mel_filters, num_frames) -> (batch_size, num_frames, num_mel_filters)
        mel_spec = mel_spec.permute(0, 2, 1)

        return mel_spec

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
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are `'pt'`, `'np'`.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled.
            do_normalize (`bool`, *optional*):
                Whether to apply per-feature zero-mean unit-variance normalization. Defaults to `self.do_normalize`
                (`False` for Sortformer).
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        do_normalize = do_normalize if do_normalize is not None else self.do_normalize

        # Convert to torch tensor
        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_sequence:
            for speech in raw_speech:
                if len(speech.shape) > 1:
                    logger.warning(
                        f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                        "We will take the mean of the channels to convert to mono."
                    )
                    speech = speech.mean(-1)

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

        # preemphasis
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
        mask = attention_mask.unsqueeze(-1)

        if do_normalize:
            # zero-mean unit-variance normalize the mel features, ignoring padding
            input_features_masked = input_features * mask
            mean = input_features_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
            mean = mean.unsqueeze(1)
            variance = ((input_features_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
            std = torch.sqrt(variance).unsqueeze(1)
            input_features = (input_features - mean) / (std + EPSILON)

        # always zero out padding frames
        input_features = input_features * mask

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


__all__ = ["SortformerFeatureExtractor"]
