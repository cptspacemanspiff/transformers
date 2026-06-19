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
r"""
Convert a NeMo Nemotron 3.5 ASR streaming checkpoint (`.nemo`) to the HuggingFace format.

Example:

```bash
python src/transformers/models/nemotron_asr/convert_nemotron_asr_nemo_to_hf.py \
    --hf_repo_id nvidia/nemotron-3.5-asr-streaming-0.6b \
    --output_dir ./nemotron-asr-hf \
    --att_context_size 56 13
```

The FastConformer encoder / RNN-T weight remapping is shared with the Parakeet conversion; this script additionally
maps the language-prompt MLP (`prompt_kernel`) and builds the prompt-dictionary into the config and processor.
"""

import argparse
import os
import re
import tarfile
import tempfile

import torch
import yaml
from tokenizers import AddedToken

from transformers import (
    NemotronAsrConfig,
    NemotronAsrEncoderConfig,
    NemotronAsrFeatureExtractor,
    NemotronAsrForRNNT,
    NemotronAsrProcessor,
    NemotronAsrTokenizer,
)
from transformers.convert_slow_tokenizer import ParakeetConverter
from transformers.utils.hub import cached_file


# NeMo -> HF weight key remapping (encoder + attention + RNN-T decoder/joint). `prompt_kernel` keeps its name.
NEMO_TO_HF_WEIGHT_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.",
    r"joint\.pred\.": r"decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": r"joint.head.",
}


def convert_key(key: str, mapping: dict) -> str:
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> dict:
    with tarfile.open(nemo_file_path, "r", encoding="utf-8") as tar:
        tar.extractall(extract_dir)

    model_files = {}
    for root, _, files in os.walk(extract_dir):
        for file in files:
            path = os.path.join(root, file)
            low = file.lower()
            if file.endswith((".ckpt", ".pt", ".pth", ".bin")) or low == "model_weights.ckpt":
                model_files["model_weights"] = path
            elif file == "model_config.yaml":
                model_files["model_config"] = path
            elif file.endswith(".model"):
                model_files["tokenizer_model_file"] = path
    for required in ("model_weights", "model_config", "tokenizer_model_file"):
        if required not in model_files:
            raise FileNotFoundError(f"Could not find {required} in {nemo_file_path}")
    return model_files


def build_encoder_config(nemo_config: dict, att_context_size) -> NemotronAsrEncoderConfig:
    enc = nemo_config["encoder"]
    return NemotronAsrEncoderConfig(
        hidden_size=enc["d_model"],
        num_hidden_layers=enc["n_layers"],
        num_attention_heads=enc["n_heads"],
        num_mel_bins=enc["feat_in"],
        conv_kernel_size=enc["conv_kernel_size"],
        subsampling_factor=enc["subsampling_factor"],
        subsampling_conv_channels=enc["subsampling_conv_channels"],
        max_position_embeddings=enc.get("pos_emb_max_len", 5000),
        dropout=enc.get("dropout", 0.1),
        dropout_positions=enc.get("dropout_emb", 0.0),
        attention_dropout=enc.get("dropout_att", 0.1),
        scale_input=enc.get("xscaling", False),
        attention_bias=enc.get("use_bias", False),
        convolution_bias=enc.get("use_bias", False),
        causal_downsampling=enc.get("causal_downsampling", True),
        conv_norm_type=enc.get("conv_norm_type", "layer_norm"),
        att_context_size=list(att_context_size),
        att_context_style=enc.get("att_context_style", "chunked_limited"),
    )


def build_config(nemo_config: dict, encoder_config: NemotronAsrEncoderConfig) -> NemotronAsrConfig:
    labels = nemo_config["joint"]["vocabulary"]
    blank_token_id = len(labels)
    vocab_size = len(labels) + 1  # +1 for the blank token appended to the tokenizer
    prednet = nemo_config["decoder"]["prednet"]
    prompt_dictionary = dict(nemo_config["model_defaults"].get("prompt_dictionary", {}))
    num_prompts = nemo_config["model_defaults"].get("num_prompts", 128)
    return NemotronAsrConfig(
        vocab_size=vocab_size,
        decoder_hidden_size=prednet.get("pred_hidden", 640),
        num_decoder_layers=prednet.get("pred_rnn_layers", 2),
        hidden_act=nemo_config["joint"]["jointnet"].get("activation", "relu"),
        max_symbols_per_step=nemo_config.get("decoding", {}).get("greedy", {}).get("max_symbols", 10),
        num_prompts=num_prompts,
        prompt_dictionary=prompt_dictionary,
        encoder_config=encoder_config.to_dict(),
        pad_token_id=blank_token_id,
        blank_token_id=blank_token_id,
    )


def convert_state_dict(model_files: dict) -> dict:
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("preprocessor.") or key.endswith(("featurizer.window", "featurizer.fb")):
            continue
        converted[convert_key(key, NEMO_TO_HF_WEIGHT_MAPPING)] = value
    return converted


def write_processor(nemo_config, model_files, output_dir):
    tokenizer_object = ParakeetConverter(model_files["tokenizer_model_file"]).converted()
    tokenizer = NemotronAsrTokenizer(tokenizer_object=tokenizer_object, clean_up_tokenization_spaces=False)
    if tokenizer.convert_tokens_to_ids("<unk>") is None:
        tokenizer.add_tokens([AddedToken("<unk>", normalized=False, special=True)])
    # RNN-T: append <blank> so it lands on id == len(labels); reuse it as pad.
    tokenizer.add_tokens([AddedToken("<blank>", normalized=False, special=True)])
    tokenizer.add_special_tokens(
        {
            "pad_token": AddedToken("<blank>", normalized=False, special=True),
            "unk_token": AddedToken("<unk>", normalized=False, special=True),
        }
    )

    pre = nemo_config["preprocessor"]
    feature_extractor = NemotronAsrFeatureExtractor(
        feature_size=pre.get("features", 128),
        sampling_rate=pre.get("sample_rate", 16000),
        win_length=int(pre.get("window_size", 0.025) * pre.get("sample_rate", 16000)),
        hop_length=int(pre.get("window_stride", 0.01) * pre.get("sample_rate", 16000)),
        n_fft=pre.get("n_fft", 512),
        do_normalize=str(pre.get("normalize", "NA")).lower() in ("per_feature", "all_features", "true"),
    )
    processor = NemotronAsrProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        prompt_dictionary=dict(nemo_config["model_defaults"].get("prompt_dictionary", {})),
        decoder_type="rnnt",
    )
    processor.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--nemo_file", default=None, help="Path to a local .nemo (skips download).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--att_context_size", type=int, nargs=2, default=[56, 13], help="[left, right] in 80ms frames")
    parser.add_argument("--push_to_repo_id", default=None)
    args = parser.parse_args()

    filepath = args.nemo_file or cached_file(args.hf_repo_id, f"{args.hf_repo_id.split('/')[-1]}.nemo")

    with tempfile.TemporaryDirectory() as tmp:
        model_files = extract_nemo_archive(filepath, tmp)
        nemo_config = yaml.load(open(model_files["model_config"]), Loader=yaml.FullLoader)

        encoder_config = build_encoder_config(nemo_config, args.att_context_size)
        config = build_config(nemo_config, encoder_config)
        print(f"Config: vocab={config.vocab_size}, num_prompts={config.num_prompts}, att={args.att_context_size}")

        converted_state_dict = convert_state_dict(model_files)

        with torch.device("meta"):
            model = NemotronAsrForRNNT(config)
        missing, unexpected = model.load_state_dict(converted_state_dict, strict=False, assign=True)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print("All weights loaded successfully!")

        if hasattr(model.config, "_name_or_path"):
            del model.config._name_or_path
        model.generation_config.decoder_start_token_id = config.blank_token_id

        write_processor(nemo_config, model_files, args.output_dir)
        model.save_pretrained(args.output_dir)
        print(f"Saved to {args.output_dir}")

        if args.push_to_repo_id:
            model.push_to_hub(args.push_to_repo_id)


if __name__ == "__main__":
    main()
