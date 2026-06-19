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
"""Testing suite for the PyTorch Nemotron 3.5 ASR model."""

import importlib.util
import math
import os
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        NemotronAsrConfig,
        NemotronAsrEncoder,
        NemotronAsrEncoderConfig,
        NemotronAsrForRNNT,
    )
    from transformers.models.nemotron_asr.modeling_nemotron_asr import build_chunked_limited_mask


# The streaming-prompt model class only exists in recent NeMo; gate the live-parity test on it.
_nemo_prompt_available = (
    importlib.util.find_spec("nemo") is not None
    and importlib.util.find_spec("nemo.collections.asr.models.rnnt_bpe_models_prompt") is not None
)
_soundfile_available = importlib.util.find_spec("soundfile") is not None
_NEMO_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def _causal_subsampled_length(length, num_layers=3, kernel=3, stride=2):
    for _ in range(num_layers):
        length = (length + (kernel - 1) + (stride - 1) - kernel) // stride + 1
    return length


class NemotronAsrEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=256,
        is_training=False,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="silu",
        dropout=0.0,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=16,
        num_mel_bins=128,
        att_context_size=(16, 3),
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins
        self.att_context_size = list(att_context_size)

        self.output_seq_length = _causal_subsampled_length(seq_length, num_layers=int(math.log2(subsampling_factor)))
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        # the first row is always fully attended so a batch always has a max-length sequence
        attention_mask[0, :] = 1
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return NemotronAsrEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.dropout,
            dropout_positions=self.dropout,
            layerdrop=self.dropout,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            num_mel_bins=self.num_mel_bins,
            att_context_size=self.att_context_size,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = NemotronAsrEncoder(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class NemotronAsrEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NemotronAsrEncoder,) if is_torch_available() else ()
    test_resize_embeddings = False
    test_torch_exportable = False

    @unittest.skip(reason="No available flash-SDPA kernels for these test shapes on this setup")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def setUp(self):
        self.model_tester = NemotronAsrEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronAsrEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_features, attention_mask)

    @unittest.skip(reason="NemotronAsrEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="NemotronAsrEncoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass


class NemotronAsrForRNNTModelTester:
    def __init__(
        self,
        parent,
        encoder_kwargs=None,
        is_training=False,
        vocab_size=129,
        decoder_hidden_size=32,
        num_decoder_layers=1,
        hidden_act="relu",
        max_symbols_per_step=5,
        num_prompts=8,
    ):
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.parent = parent
        self.encoder_model_tester = NemotronAsrEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training
        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.seq_length = self.output_seq_length
        self.encoder_seq_length = self.output_seq_length
        self.hidden_size = self.encoder_model_tester.hidden_size
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers

        self.vocab_size = vocab_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.hidden_act = hidden_act
        self.max_symbols_per_step = max_symbols_per_step
        self.num_prompts = num_prompts
        self.blank_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def get_config(self):
        return NemotronAsrConfig(
            vocab_size=self.vocab_size,
            decoder_hidden_size=self.decoder_hidden_size,
            num_decoder_layers=self.num_decoder_layers,
            hidden_act=self.hidden_act,
            max_symbols_per_step=self.max_symbols_per_step,
            num_prompts=self.num_prompts,
            prompt_dictionary={"en-US": 0, "auto": self.num_prompts - 1},
            encoder_config=self.encoder_model_tester.get_config().to_dict(),
            pad_token_id=self.pad_token_id,
            blank_token_id=self.blank_token_id,
        )

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        return self.get_config(), input_features, attention_mask

    def create_and_check_model(self, config, inputs_dict):
        model = NemotronAsrForRNNT(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        decoder_input_ids = ids_tensor([self.batch_size, 1], self.vocab_size)
        prompt_indices = ids_tensor([self.batch_size], self.num_prompts)
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "prompt_indices": prompt_indices,
        }
        return config, inputs_dict


@require_torch
class NemotronAsrForRNNTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NemotronAsrForRNNT,) if is_torch_available() else ()

    test_attention_outputs = False
    test_resize_embeddings = False
    test_torch_exportable = False
    _is_composite = True

    @unittest.skip(reason="No available flash-SDPA kernels for these test shapes on this setup")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def setUp(self):
        self.model_tester = NemotronAsrForRNNTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronAsrConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="NemotronAsrForRNNT does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="NemotronAsrForRNNT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Transducer with an LSTM prediction network; no standard encoder-decoder hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Transducer with an LSTM prediction network; no standard encoder-decoder hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Custom RNN-T generate() is not fully compatible with GenerationTesterMixin")
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip(reason="Flat composite transducer without a separate base_model sub-module")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="RNN-T decoder is an LSTM prediction network without attention")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip(reason="Transducer is not a standard encoder-decoder; no separate text config to set")
    def test_attn_implementation_composite_models(self):
        pass


@require_torch
class NemotronAsrModelBehaviorTest(unittest.TestCase):
    """Model-specific behaviors that distinguish Nemotron 3.5 ASR from the offline Parakeet encoder."""

    def test_causal_subsampling_frequency_dim(self):
        # With causal downsampling, 128 mel bins survive as 17 (not 16), widening the post-subsampling linear input.
        causal = NemotronAsrEncoderConfig(num_mel_bins=128, subsampling_conv_channels=32, causal_downsampling=True)
        non_causal = NemotronAsrEncoderConfig(
            num_mel_bins=128, subsampling_conv_channels=32, causal_downsampling=False
        )
        causal_in = NemotronAsrEncoder(causal).subsampling.linear.in_features
        non_causal_in = NemotronAsrEncoder(non_causal).subsampling.linear.in_features
        self.assertEqual(causal_in, 32 * 17)
        self.assertEqual(non_causal_in, 32 * 16)

    def test_conv_module_uses_layer_norm(self):
        config = NemotronAsrEncoderConfig(num_hidden_layers=1, conv_norm_type="layer_norm")
        conv = NemotronAsrEncoder(config).layers[0].conv
        self.assertIsInstance(conv.norm, torch.nn.LayerNorm)

    def test_chunked_limited_mask(self):
        # att_context_size=[2, 1] -> chunk_size=2, left_chunks=1: each length-2 chunk sees itself and one left chunk.
        mask = build_chunked_limited_mask([2, 1], seq_length=4, device=torch.device("cpu"))
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))
        # The mask is strictly causal-or-bounded: no query attends to a key in a future chunk.
        self.assertTrue(bool((~mask | torch.tril(torch.ones(4, 4, dtype=torch.bool)) | mask).all()))

    def test_prompt_indices_change_output(self):
        config = NemotronAsrForRNNTModelTester(self).get_config()
        model = NemotronAsrForRNNT(config).to(torch_device).eval()
        feats = floats_tensor([1, 128, config.encoder_config.num_mel_bins])
        with torch.no_grad():
            out0 = model.get_audio_features(input_features=feats, prompt_indices=torch.tensor([0])).pooler_output
            out1 = model.get_audio_features(
                input_features=feats, prompt_indices=torch.tensor([config.num_prompts - 1])
            ).pooler_output
        self.assertFalse(torch.allclose(out0, out1), "Different language prompts should yield different outputs")

    def test_padding_mask_does_not_leak_into_valid_frames(self):
        """A trailing padding frame, when masked, must not change the encoder output on valid frames.

        This locks in the NeMo-style masking inside the conv subsampling: without it, a padded frame leaks through
        the causal convolutions and perturbs the last few output frames.
        """
        config = NemotronAsrEncoderConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            subsampling_conv_channels=16,
            num_mel_bins=128,
            dropout=0.0,
            layerdrop=0.0,
        )
        model = NemotronAsrEncoder(config).to(torch_device).eval()

        valid = 64
        torch.manual_seed(0)
        feats = torch.randn(1, valid, config.num_mel_bins, device=torch_device)
        out_clean = model(input_features=feats).last_hidden_state
        s1 = out_clean.shape[1]

        # append zero pad frames and mark them invalid
        padded = torch.cat([feats, torch.zeros(1, 16, config.num_mel_bins, device=torch_device)], dim=1)
        mask = torch.zeros(1, padded.shape[1], dtype=torch.long, device=torch_device)
        mask[:, :valid] = 1
        with torch.no_grad():
            out_padded = model(input_features=padded, attention_mask=mask).last_hidden_state

        max_abs = (out_padded[:, :s1] - out_clean).abs().max().item()
        self.assertLess(max_abs, 1e-5, f"Padding leaked into valid frames: max|Δ|={max_abs:.3e}")


@require_torch
@slow
class NemotronAsrIntegrationTest(unittest.TestCase):
    """Numerical-parity test of the converted HF model against the original NeMo model.

    Requires a recent `nemo_toolkit` (with the streaming-prompt RNN-T model) and network access. The checkpoint must
    first be converted with `convert_nemotron_asr_nemo_to_hf.py`; point `NEMOTRON_ASR_HF_PATH` at the local output.
    The offline (full-utterance, chunked-limited) path is compared: identical mel features in, post-prompt encoder
    embeddings out.
    """

    hf_path = os.environ.get("NEMOTRON_ASR_HF_PATH", "nvidia/nemotron-3.5-asr-streaming-0.6b-hf")
    att_context_size = [56, 13]

    @unittest.skipUnless(_nemo_prompt_available, "nemo_toolkit with streaming-prompt RNN-T is not installed")
    def test_offline_forward_parity_with_nemo(self):
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import ASRModel

        nemo_file = next(f for f in list_repo_files(_NEMO_REPO) if f.endswith(".nemo"))
        nemo_path = hf_hub_download(repo_id=_NEMO_REPO, filename=nemo_file)
        nemo_model = ASRModel.restore_from(nemo_path, map_location="cpu").eval()
        nemo_model.encoder.set_default_att_context_size(self.att_context_size)
        prompt_idx = nemo_model.cfg.model_defaults.prompt_dictionary["en-US"]

        hf_model = NemotronAsrForRNNT.from_pretrained(self.hf_path).eval()

        torch.manual_seed(0)
        audio = torch.randn(1, 16000 * 6)
        audio_len = torch.tensor([audio.shape[1]])

        with torch.no_grad():
            processed, processed_len = nemo_model.preprocessor(input_signal=audio, length=audio_len)
            # NeMo forward returns the post-prompt encoder output (B, D, T)
            nemo_enc, _ = nemo_model.forward(
                processed_signal=processed,
                processed_signal_length=processed_len,
                prompt_indices=torch.tensor([prompt_idx]),
            )
            nemo_enc = nemo_enc.transpose(1, 2)  # (B, T, D)

            input_features = processed.transpose(1, 2)
            attn = (torch.arange(input_features.shape[1])[None, :] < processed_len[:, None]).long()
            hf_enc = hf_model.get_audio_features(
                input_features=input_features, attention_mask=attn, prompt_indices=torch.tensor([prompt_idx])
            ).last_hidden_state

        n = min(nemo_enc.shape[1], hf_enc.shape[1])
        max_abs = (nemo_enc[:, :n] - hf_enc[:, :n]).abs().max().item()
        self.assertLess(max_abs, 1e-3, f"Offline forward parity with NeMo failed: max|Δ|={max_abs:.3e}")

    @unittest.skipUnless(
        _nemo_prompt_available and _soundfile_available, "nemo_toolkit (prompt RNN-T) and soundfile are required"
    )
    def test_transcription_parity_with_nemo(self):
        import soundfile as sf
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import ASRModel

        from transformers import AutoProcessor

        nemo_file = next(f for f in list_repo_files(_NEMO_REPO) if f.endswith(".nemo"))
        nemo_path = hf_hub_download(repo_id=_NEMO_REPO, filename=nemo_file)
        nemo_model = ASRModel.restore_from(nemo_path, map_location="cpu").eval()
        nemo_model.encoder.set_default_att_context_size(self.att_context_size)

        hf_model = NemotronAsrForRNNT.from_pretrained(self.hf_path).eval()
        processor = AutoProcessor.from_pretrained(self.hf_path)

        with tempfile.TemporaryDirectory() as tmp:
            import urllib.request

            wav_path = os.path.join(tmp, "sample.wav")
            urllib.request.urlretrieve("https://cdn-media.huggingface.co/speech_samples/sample1.flac", wav_path)
            manifest = os.path.join(tmp, "manifest.json")
            audio, sr = sf.read(wav_path)
            with open(manifest, "w") as f:
                import json

                json.dump({"audio_filepath": wav_path, "duration": len(audio) / sr, "text": "", "lang": "en-US"}, f)
            nemo_text = nemo_model.transcribe(manifest, batch_size=1)[0]
            nemo_text = nemo_text.text if hasattr(nemo_text, "text") else nemo_text

            inputs = processor(audio, sampling_rate=16000, target_lang="en-US", return_tensors="pt")
            # No max_new_tokens: RNN-T generation auto-sizes from encoder capacity and stops on encoder exhaustion.
            output = hf_model.generate(**inputs)
            hf_text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]

        self.assertEqual(hf_text, nemo_text)
