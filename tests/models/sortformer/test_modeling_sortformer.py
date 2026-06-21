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
"""Testing suite for the PyTorch Sortformer model."""

import importlib.util
import json
import os
import tempfile
import unittest
import urllib.request

from transformers import SortformerConfig, is_torch_available
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import SortformerFeatureExtractor, SortformerForAudioFrameClassification, SortformerModel


_nemo_available = importlib.util.find_spec("nemo") is not None
_librosa_available = importlib.util.find_spec("librosa") is not None
_soundfile_available = importlib.util.find_spec("soundfile") is not None
_STREAMING_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"
# Canonical NeMo 2-speaker diarization test utterance (16 kHz, ~5 s).
_AN4_DIARIZE_TEST_URL = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
# Committed synthetic 4-speaker overlapping fixture (see tests/fixtures/sortformer/).
_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "sortformer")


class SortformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=1024,
        is_training=True,
        # transformer encoder
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_speakers=4,
        # fastconformer encoder
        encoder_hidden_size=64,
        encoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        encoder_intermediate_size=128,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        num_mel_bins=128,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_speakers = num_speakers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins

        self.output_seq_length = seq_length // subsampling_factor
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def get_config(self):
        encoder_config = ParakeetEncoderConfig(
            hidden_size=self.encoder_hidden_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            num_attention_heads=self.encoder_num_attention_heads,
            intermediate_size=self.encoder_intermediate_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            num_mel_bins=self.num_mel_bins,
            dropout=0.0,
            dropout_positions=0.0,
            layerdrop=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
        )
        return SortformerConfig(
            encoder_config=encoder_config,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            num_speakers=self.num_speakers,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_layer_dropout=0.0,
            head_dropout=0.0,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()
        return config, input_features, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features, "attention_mask": attention_mask}
        return config, inputs_dict

    def create_and_check_model(self, config, input_features, attention_mask):
        model = SortformerModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )

    def create_and_check_for_audio_frame_classification(self, config, input_features, attention_mask):
        model = SortformerForAudioFrameClassification(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.output_seq_length, config.num_speakers))


@require_torch
class SortformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SortformerModel, SortformerForAudioFrameClassification) if is_torch_available() else ()
    pipeline_model_mapping = {}
    test_resize_embeddings = False
    test_pruning = False
    test_headmasking = False
    test_attention_outputs = False
    test_torch_exportable = False
    _is_composite = True

    def setUp(self):
        self.model_tester = SortformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SortformerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_audio_frame_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_audio_frame_classification(*config_and_inputs)

    @unittest.skip(reason="Sortformer does not use input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Sortformer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Sortformer does not expose intermediate hidden_states in the standard form")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Sortformer does not expose hidden_states/attentions in the standard form")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Per-frame multi-label diarization targets are not generated by the common test harness")
    def test_training(self):
        pass

    @unittest.skip(reason="Per-frame multi-label diarization targets are not generated by the common test harness")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Per-frame multi-label diarization targets are not generated by the common test harness")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Per-frame multi-label diarization targets are not generated by the common test harness")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Per-frame multi-label diarization targets are not generated by the common test harness")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@require_torch
class SortformerStreamingConformerTest(unittest.TestCase):
    """Cache-aware streaming of the FastConformer encoder (causal convs + growing attention cache)."""

    def _build_model(self):
        config = SortformerModelTester(self).get_config()
        return config, SortformerForAudioFrameClassification(config).to(torch_device).eval()

    def test_streaming_diarize_shape(self):
        config, model = self._build_model()
        features = floats_tensor([1, 25 + 8 * 5, config.encoder_config.num_mel_bins]).to(torch_device)
        with torch.no_grad():
            logits = model.streaming_diarize(features, right_context=7)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], config.num_speakers)

    def test_streaming_matches_single_pass(self):
        """Streaming in small chunks must equal one cache-aware pass over the whole sequence (cache correctness)."""
        config, model = self._build_model()
        torch.manual_seed(0)
        features = torch.randn(1, 25 + 8 * 6, config.encoder_config.num_mel_bins, device=torch_device)
        with torch.no_grad():
            chunked = model.sortformer.streaming_encode(features, right_context=7, chunk_size=8)
            one_shot = model.sortformer.streaming_encode(features, right_context=7, chunk_size=100_000)
        self.assertEqual(chunked.shape, one_shot.shape)
        self.assertLess((chunked - one_shot).abs().max().item(), 1e-4)

    def test_offline_conformer_forward_matches_encoder(self):
        """The single `_conformer_forward(cache=None, causal=False)` must reproduce `self.encoder(...)` exactly, so the
        offline output is numerically unchanged by routing through the shared cache-aware forward."""
        config, model = self._build_model()
        torch.manual_seed(0)
        sortformer = model.sortformer
        features = torch.randn(2, 8 * 40, config.encoder_config.num_mel_bins, device=torch_device)
        attention_mask = torch.ones(features.shape[0], features.shape[1], dtype=torch.long, device=torch_device)
        attention_mask[1, 8 * 30 :] = 0
        with torch.no_grad():
            ref = sortformer.encoder(features, attention_mask=attention_mask)
            hidden_states, output_mask, updated_cache = sortformer._conformer_forward(
                features, attention_mask=attention_mask, cache=None, causal=False
            )
        self.assertIsNone(updated_cache)
        self.assertEqual(hidden_states.shape, ref.last_hidden_state.shape)
        self.assertLess((hidden_states - ref.last_hidden_state).abs().max().item(), 1e-5)
        self.assertTrue(torch.equal(output_mask, ref.attention_mask))


@require_torch
@slow
class SortformerIntegrationTest(unittest.TestCase):
    """Numerical-parity test of the converted HF model against the original NeMo model.

    Requires `nemo_toolkit` to be installed and network access to download the NeMo checkpoint. The model must first
    be converted with `convert_sortformer_nemo_to_hf.py`. The offline (full-sequence) forward path is compared, i.e.
    FastConformer encoder + Transformer encoder + speaker head fed identical mel features.
    """

    # Override with the env var to validate a locally-converted checkpoint before it is published to the Hub.
    hf_path = os.environ.get("SORTFORMER_HF_PATH", "nvidia/diar_streaming_sortformer_4spk-v2-hf")

    @unittest.skipUnless(_nemo_available, "nemo_toolkit is not installed")
    def test_offline_forward_parity_with_nemo(self):
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import SortformerEncLabelModel

        nemo_file = next(f for f in list_repo_files(_STREAMING_REPO) if f.endswith(".nemo"))
        nemo_path = hf_hub_download(repo_id=_STREAMING_REPO, filename=nemo_file)

        nemo_model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu").eval()
        nemo_model.streaming_mode = False  # force offline full-sequence forward path

        hf_model = SortformerForAudioFrameClassification.from_pretrained(self.hf_path).eval()

        torch.manual_seed(0)
        audio = torch.randn(1, 16000 * 8)
        audio_len = torch.tensor([audio.shape[1]])

        with torch.no_grad():
            processed_signal, processed_len = nemo_model.process_signal(audio, audio_len)
            processed_signal = processed_signal[:, :, : processed_len.max()]
            emb_seq, emb_len = nemo_model.frontend_encoder(processed_signal, processed_len)
            nemo_preds = nemo_model.forward_infer(emb_seq, emb_len)

            input_features = processed_signal.transpose(1, 2)
            attn = (torch.arange(input_features.shape[1])[None, :] < processed_len[:, None]).long()
            hf_preds = hf_model(input_features=input_features, attention_mask=attn).logits.sigmoid()

        n = min(nemo_preds.shape[1], hf_preds.shape[1])
        max_abs = (nemo_preds[:, :n] - hf_preds[:, :n]).abs().max().item()
        self.assertLess(max_abs, 1e-3, f"Offline forward parity with NeMo failed: max|Δ|={max_abs:.3e}")

    @unittest.skipUnless(
        _nemo_available and _librosa_available and _soundfile_available,
        "nemo_toolkit, librosa and soundfile are required",
    )
    def test_offline_pipeline_parity_with_nemo_real_audio(self):
        """End-to-end parity on real speech: the full HF pipeline (feature extractor + model) must match NeMo's full
        offline forward (``process_signal`` -> ``frontend_encoder`` -> ``forward_infer``) on a real 2-speaker
        recording. This additionally validates the [`SortformerFeatureExtractor`] and the offline waveform
        peak-normalization, which the random-tensor test bypasses by feeding identical pre-computed features.
        """
        import soundfile as sf
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import SortformerEncLabelModel

        nemo_file = next(f for f in list_repo_files(_STREAMING_REPO) if f.endswith(".nemo"))
        nemo_path = hf_hub_download(repo_id=_STREAMING_REPO, filename=nemo_file)

        nemo_model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu").eval()
        nemo_model.streaming_mode = False  # force offline full-sequence forward path

        hf_model = SortformerForAudioFrameClassification.from_pretrained(self.hf_path).eval()
        feature_extractor = SortformerFeatureExtractor.from_pretrained(self.hf_path)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "an4_diarize_test.wav")
            urllib.request.urlretrieve(_AN4_DIARIZE_TEST_URL, wav_path)
            audio, sampling_rate = sf.read(wav_path)
        self.assertEqual(sampling_rate, 16000)
        audio = torch.tensor(audio, dtype=torch.float32)[None, :]
        audio_len = torch.tensor([audio.shape[1]])

        with torch.no_grad():
            # NeMo: full offline pipeline (process_signal applies the 1/(max+eps) peak-normalization).
            processed_signal, processed_len = nemo_model.process_signal(audio, audio_len)
            processed_signal = processed_signal[:, :, : processed_len.max()]
            emb_seq, emb_len = nemo_model.frontend_encoder(processed_signal, processed_len)
            nemo_preds = nemo_model.forward_infer(emb_seq, emb_len)

            # HF: replicate the offline peak-normalization, then run the feature extractor + model.
            audio_norm = (1.0 / (audio.max() + nemo_model.eps)) * audio
            inputs = feature_extractor(audio_norm[0].numpy(), sampling_rate=16000, return_tensors="pt")
            hf_preds = hf_model(**inputs).logits.sigmoid()

        n = min(nemo_preds.shape[1], hf_preds.shape[1])
        max_abs = (nemo_preds[:, :n] - hf_preds[:, :n]).abs().max().item()
        # Bit-exact in practice (~5e-7); the small tolerance only covers float32 STFT/mel rounding in silent bins.
        self.assertLess(max_abs, 1e-4, f"Real-audio pipeline parity with NeMo failed: max|Δ|={max_abs:.3e}")

        # Sanity check: the model should detect 2 speakers on this 2-speaker recording.
        active_speakers = [s for s in range(hf_preds.shape[-1]) if (hf_preds[0, :, s] > 0.5).sum().item() > 3]
        self.assertEqual(len(active_speakers), 2, f"Expected 2 active speakers, got {active_speakers}")

    @unittest.skipUnless(
        _nemo_available and _librosa_available and _soundfile_available,
        "nemo_toolkit, librosa and soundfile are required",
    )
    def test_offline_pipeline_parity_with_nemo_4spk_overlap(self):
        """Parity + diarization sanity on the committed synthetic 4-speaker overlapping fixture.

        The fixture (``tests/fixtures/sortformer/mixture_4spk_overlap.{wav,json}``) has four speakers (~8s each) with
        widely-spaced onsets so that at most two overlap at once, and one speaker mixed at a lower level. This checks
        that (1) the full HF pipeline matches NeMo's offline forward, and (2) the model produces overlapping
        multi-speaker output.
        """
        import soundfile as sf
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import SortformerEncLabelModel

        with open(os.path.join(_FIXTURES_DIR, "mixture_4spk_overlap.json")) as f:
            meta = json.load(f)
        self.assertLessEqual(meta["max_simultaneous_speakers"], 2)
        audio, sampling_rate = sf.read(os.path.join(_FIXTURES_DIR, meta["audio_file"]))
        self.assertEqual(sampling_rate, meta["sampling_rate"])

        nemo_file = next(f for f in list_repo_files(_STREAMING_REPO) if f.endswith(".nemo"))
        nemo_path = hf_hub_download(repo_id=_STREAMING_REPO, filename=nemo_file)
        nemo_model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu").eval()
        nemo_model.streaming_mode = False  # force offline full-sequence forward path

        hf_model = SortformerForAudioFrameClassification.from_pretrained(self.hf_path).eval()
        feature_extractor = SortformerFeatureExtractor.from_pretrained(self.hf_path)

        audio = torch.tensor(audio, dtype=torch.float32)[None, :]
        audio_len = torch.tensor([audio.shape[1]])

        with torch.no_grad():
            processed_signal, processed_len = nemo_model.process_signal(audio, audio_len)
            processed_signal = processed_signal[:, :, : processed_len.max()]
            emb_seq, emb_len = nemo_model.frontend_encoder(processed_signal, processed_len)
            nemo_preds = nemo_model.forward_infer(emb_seq, emb_len)

            audio_norm = (1.0 / (audio.max() + nemo_model.eps)) * audio
            inputs = feature_extractor(audio_norm[0].numpy(), sampling_rate=16000, return_tensors="pt")
            hf_preds = hf_model(**inputs).logits.sigmoid()

        n = min(nemo_preds.shape[1], hf_preds.shape[1])
        max_abs = (nemo_preds[:, :n] - hf_preds[:, :n]).abs().max().item()
        self.assertLess(max_abs, 1e-4, f"4-speaker overlap pipeline parity with NeMo failed: max|Δ|={max_abs:.3e}")

        # The model must produce genuine overlapping (multi-label) output on this overlapping mixture.
        active = (hf_preds[0] > 0.5).sum(dim=0)  # per-speaker active-frame counts
        num_active = int((active > 3).sum().item())
        self.assertGreaterEqual(num_active, 2, f"Expected at least 2 active speakers, got {num_active}")
        overlap_frames = int(((hf_preds[0] > 0.5).sum(dim=-1) >= 2).sum().item())
        self.assertGreater(overlap_frames, 0, "Expected the model to produce overlapping (>=2 speaker) frames")

    @unittest.skipUnless(
        _nemo_available and _soundfile_available,
        "nemo_toolkit and soundfile are required",
    )
    def test_streaming_parity_with_nemo(self):
        """Numerical parity of the HF streaming path (`diarize_streaming`, the Arrival-Order Speaker Cache) against
        NeMo's synchronous `forward_streaming`.

        Identical mel features are fed to both so this isolates the chunked encoder + AOSC update/compression logic.
        The 27s 4-speaker fixture spans two chunks, so the speaker cache overflows `spkcache_len` and is compressed at
        least once. Unlike the offline path, streaming does **not** peak-normalize the waveform.
        """
        import soundfile as sf
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import SortformerEncLabelModel

        audio, sampling_rate = sf.read(os.path.join(_FIXTURES_DIR, "mixture_4spk_overlap.wav"))
        self.assertEqual(sampling_rate, 16000)

        nemo_file = next(f for f in list_repo_files(_STREAMING_REPO) if f.endswith(".nemo"))
        nemo_path = hf_hub_download(repo_id=_STREAMING_REPO, filename=nemo_file)
        nemo_model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu").eval()
        nemo_model.streaming_mode = True  # enable the chunked streaming path

        hf_model = SortformerForAudioFrameClassification.from_pretrained(self.hf_path).eval()

        audio = torch.tensor(audio, dtype=torch.float32)[None, :]
        audio_len = torch.tensor([audio.shape[1]])

        with torch.no_grad():
            # NeMo streaming: process_signal does NOT peak-normalize in streaming_mode.
            processed_signal, processed_len = nemo_model.process_signal(audio, audio_len)
            processed_signal = processed_signal[:, :, : processed_len.max()]
            nemo_preds = nemo_model.forward_streaming(processed_signal, processed_len)

            # HF: feed identical features into the streaming wrapper.
            input_features = processed_signal.transpose(1, 2)
            attn = (torch.arange(input_features.shape[1])[None, :] < processed_len[:, None]).long()
            hf_preds = hf_model.diarize_streaming(input_features, attention_mask=attn)

        self.assertEqual(hf_preds.shape, nemo_preds.shape)
        max_abs = (nemo_preds - hf_preds).abs().max().item()
        # Bit-exact in practice (~6e-7); tolerance only covers float32 rounding in the encoder/AOSC tensor ops.
        self.assertLess(max_abs, 1e-4, f"Streaming parity with NeMo failed: max|Δ|={max_abs:.3e}")

        # The cache must have been compressed at least once (two chunks, cache cap 188 < 338 total frames).
        self.assertGreater(nemo_preds.shape[1], hf_model.config.spkcache_len)
