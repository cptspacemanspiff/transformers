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
Convert a NeMo Sortformer diarization checkpoint (`.nemo`) to the HuggingFace format.

Example:

```bash
python src/transformers/models/sortformer/convert_sortformer_nemo_to_hf.py \
    --hf_repo_id nvidia/diar_streaming_sortformer_4spk-v2 \
    --output_dir ./sortformer-hf
```

The FastConformer encoder weight remapping is shared with the Parakeet conversion (the Sortformer encoder *is* a
NeMo ConformerEncoder, i.e. a `ParakeetEncoder`). The Transformer-encoder and speaker-head remappings are
Sortformer-specific. Sortformer module submodules were named to match the NeMo originals, so the mapping is a small
set of prefix rewrites.
"""

import argparse
import os
import re
import tarfile
import tempfile

import torch
import yaml

from transformers import (
    SortformerConfig,
    SortformerFeatureExtractor,
    SortformerForAudioFrameClassification,
)
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig


# NeMo -> HF weight-name mapping. Applied in order with re.sub, so put specific rules before the generic
# `^encoder\.` prefix rewrite.
NEMO_TO_HF_WEIGHT_MAPPING = {
    # --- FastConformer encoder (shared with Parakeet) ---
    r"^encoder\.pre_encode\.conv\.": r"sortformer.encoder.subsampling.layers.",
    r"^encoder\.pre_encode\.out\.": r"sortformer.encoder.subsampling.linear.",
    r"^encoder\.pos_enc\.": r"sortformer.encoder.encode_positions.",
    r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"sortformer.encoder.layers.\1.conv.norm.",
    # --- Transformer encoder (Sortformer-specific) ---
    r"^transformer_encoder\.layers\.(\d+)\.first_sub_layer\.": r"sortformer.transformer_encoder.layers.\1.self_attn.",
    r"^transformer_encoder\.layers\.(\d+)\.second_sub_layer\.": r"sortformer.transformer_encoder.layers.\1.feed_forward.",
    r"^transformer_encoder\.layers\.(\d+)\.layer_norm_": r"sortformer.transformer_encoder.layers.\1.layer_norm_",
    # --- Sortformer projection + speaker head ---
    r"^sortformer_modules\.encoder_proj\.": r"sortformer.encoder_proj.",
    r"^sortformer_modules\.first_hidden_to_hidden\.": r"speaker_head.first_hidden_to_hidden.",
    r"^sortformer_modules\.single_hidden_to_spks\.": r"speaker_head.single_hidden_to_spks.",
    # --- conformer attention renames (only fire inside encoder.* keys) ---
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
    # --- finally prefix any remaining bare encoder.* keys ---
    r"^encoder\.": r"sortformer.encoder.",
}

# NeMo prefixes whose weights are not part of the HF model.
# `sortformer_modules.hidden_to_spks` is an unused legacy head in NeMo (the active forward path uses
# `single_hidden_to_spks`), so it is intentionally dropped.
SKIP_PREFIXES = (
    "preprocessor.",
    "loss.",
    "spec_augmentation.",
    "sortformer_modules.hidden_to_spks.",
    "_",
)


def convert_key(key: str, mapping: dict) -> str:
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> dict:
    with tarfile.open(nemo_file_path, "r") as tar:
        tar.extractall(extract_dir)

    files = {}
    for root, _, filenames in os.walk(extract_dir):
        for fn in filenames:
            path = os.path.join(root, fn)
            if fn.endswith(".ckpt") or fn == "model_weights.ckpt":
                files["model_weights"] = path
            elif fn.endswith(".yaml") or fn == "model_config.yaml":
                files["model_config"] = path
    if "model_weights" not in files:
        raise FileNotFoundError(f"No model weights (.ckpt) found inside {nemo_file_path}")
    if "model_config" not in files:
        raise FileNotFoundError(f"No model_config.yaml found inside {nemo_file_path}")
    return files


def build_config(nemo_config: dict) -> SortformerConfig:
    enc = nemo_config["encoder"]
    tf = nemo_config["transformer_encoder"]
    sm = nemo_config["sortformer_modules"]
    pre = nemo_config["preprocessor"]

    ff_expansion = enc.get("ff_expansion_factor", 4)
    encoder_config = ParakeetEncoderConfig(
        hidden_size=enc["d_model"],
        num_hidden_layers=enc["n_layers"],
        num_attention_heads=enc["n_heads"],
        intermediate_size=enc["d_model"] * ff_expansion,
        hidden_act="silu",
        conv_kernel_size=enc["conv_kernel_size"],
        subsampling_factor=enc["subsampling_factor"],
        subsampling_conv_channels=enc["subsampling_conv_channels"],
        num_mel_bins=pre["features"],
        scale_input=enc.get("xscaling", True),
        max_position_embeddings=enc.get("pos_emb_max_len", 5000),
        dropout=enc.get("dropout", 0.1),
        attention_dropout=enc.get("dropout_att", 0.1),
    )

    config = SortformerConfig(
        encoder_config=encoder_config,
        hidden_size=tf["hidden_size"],
        num_hidden_layers=tf["num_layers"],
        num_attention_heads=tf["num_attention_heads"],
        intermediate_size=tf["inner_size"],
        hidden_act=tf.get("hidden_act", "relu"),
        hidden_dropout=tf.get("ffn_dropout", 0.5),
        attention_dropout=tf.get("attn_score_dropout", 0.5),
        attention_layer_dropout=tf.get("attn_layer_dropout", 0.5),
        head_dropout=sm.get("dropout_rate", 0.5),
        pre_ln=tf.get("pre_ln", False),
        pre_ln_final_layer_norm=tf.get("pre_ln_final_layer_norm", True),
        num_speakers=nemo_config.get("max_num_of_spks", sm.get("num_spks", 4)),
    )
    return config


def convert_state_dict(nemo_state_dict: dict) -> dict:
    converted = {}
    skipped = []
    for key, value in nemo_state_dict.items():
        if key.startswith(SKIP_PREFIXES):
            skipped.append(key)
            continue
        converted[convert_key(key, NEMO_TO_HF_WEIGHT_MAPPING)] = value
    if skipped:
        print(f"Skipped {len(skipped)} non-model keys (preprocessor/loss/etc.)")
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_path", default=None, help="Path to a local .nemo file.")
    parser.add_argument(
        "--hf_repo_id",
        default="nvidia/diar_streaming_sortformer_4spk-v2",
        help="HF Hub repo id to download the .nemo from if --nemo_path is not given.",
    )
    parser.add_argument("--nemo_filename", default=None, help="Name of the .nemo file within the repo.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub", default=None, help="Optional target repo id to push the converted model to.")
    args = parser.parse_args()

    nemo_path = args.nemo_path
    if nemo_path is None:
        from huggingface_hub import hf_hub_download, list_repo_files

        filename = args.nemo_filename
        if filename is None:
            nemo_files = [f for f in list_repo_files(args.hf_repo_id) if f.endswith(".nemo")]
            if not nemo_files:
                raise FileNotFoundError(f"No .nemo file found in repo {args.hf_repo_id}")
            filename = nemo_files[0]
        print(f"Downloading {filename} from {args.hf_repo_id}")
        nemo_path = hf_hub_download(repo_id=args.hf_repo_id, filename=filename)

    with tempfile.TemporaryDirectory() as tmp:
        files = extract_nemo_archive(nemo_path, tmp)
        with open(files["model_config"]) as f:
            nemo_config = yaml.safe_load(f)
        nemo_state_dict = torch.load(files["model_weights"], map_location="cpu", weights_only=True)

    config = build_config(nemo_config)
    print(config)

    converted_state_dict = convert_state_dict(nemo_state_dict)

    model = SortformerForAudioFrameClassification(config)
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    # ParakeetEncoder may register inv_freq / rotary buffers that are not in the NeMo checkpoint; report anything else.
    real_missing = [k for k in missing if "inv_freq" not in k and "rotary" not in k]
    if real_missing:
        raise RuntimeError(f"Missing keys when loading converted weights:\n{real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading converted weights:\n{unexpected}")
    print(f"Loaded weights. ({len(missing)} buffer keys auto-initialized, {len(unexpected)} unexpected)")

    pre = nemo_config["preprocessor"]
    feature_extractor = SortformerFeatureExtractor(
        feature_size=pre["features"],
        sampling_rate=pre["sample_rate"],
        hop_length=int(pre["window_stride"] * pre["sample_rate"]),
        win_length=int(pre["window_size"] * pre["sample_rate"]),
        n_fft=pre.get("n_fft", 512),
        preemphasis=pre.get("preemph", 0.97),
        do_normalize=(pre.get("normalize", "NA") not in ("NA", None, "none")),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print(f"Saved converted model + feature extractor to {args.output_dir}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        feature_extractor.push_to_hub(args.push_to_hub)
        print(f"Pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
