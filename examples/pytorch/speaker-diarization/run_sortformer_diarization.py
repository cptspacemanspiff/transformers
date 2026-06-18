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
r"""
Run Sortformer speaker diarization on an audio file and plot the result.

Sortformer is an end-to-end diarization model that emits, for every ~80 ms output frame, an independent probability
for each of up to `num_speakers` speakers being active (multi-label / sigmoid), so it naturally handles overlapping
speech. This example runs the (offline) forward pass and renders a diarization figure.

Example:

```bash
python examples/pytorch/speaker-diarization/run_sortformer_diarization.py \
    --model nvidia/diar_streaming_sortformer_4spk-v2-hf \
    --audio tests/fixtures/sortformer/mixture_4spk_overlap.wav \
    --output diarization.png
```

Requires: `pip install soundfile librosa matplotlib`.
"""

import argparse
import json
import os

import matplotlib


matplotlib.use("Agg")  # headless backend so the script runs without a display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sortformer diarization and plot speaker activity.")
    parser.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2-hf", help="HF model id or path.")
    parser.add_argument("--audio", required=True, help="Path to a mono 16 kHz wav file.")
    parser.add_argument("--output", default="diarization.png", help="Where to write the plot.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for activity.")
    parser.add_argument(
        "--no-peak-normalize",
        action="store_true",
        help="Disable the 1/(max+eps) waveform peak-normalization used by the offline model.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run chunked streaming diarization (Arrival-Order Speaker Cache) instead of the offline forward pass. "
        "The streaming path does not peak-normalize the waveform.",
    )
    return parser.parse_args()


def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def probs_to_segments(active, frame_stride):
    """Convert a boolean per-frame activity vector to a list of (start_sec, end_sec) segments."""
    segments, start = [], None
    for i, is_active in enumerate(list(active) + [False]):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            segments.append((start * frame_stride, i * frame_stride))
            start = None
    return segments


def main():
    args = parse_args()

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioFrameClassification.from_pretrained(args.model).eval()

    audio = load_audio(args.audio, feature_extractor.sampling_rate)
    # The offline forward path peak-normalizes the waveform before feature extraction; the streaming path does not.
    if not args.no_peak_normalize and not args.streaming:
        audio = audio / (np.abs(audio).max() + 1e-3)

    inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    with torch.no_grad():
        if args.streaming:
            # Chunked streaming inference (Arrival-Order Speaker Cache); returns sigmoid probabilities directly.
            probs = model.diarize_streaming(inputs.input_features, attention_mask=inputs.attention_mask)[0].numpy()
        else:
            # logits: (batch, num_frames, num_speakers); probabilities via sigmoid (multi-label).
            probs = model(**inputs).logits.sigmoid()[0].numpy()

    num_frames, num_speakers = probs.shape
    # Each output frame spans subsampling_factor * hop_length / sampling_rate seconds (~80 ms).
    frame_stride = (
        model.config.encoder_config.subsampling_factor * feature_extractor.hop_length / feature_extractor.sampling_rate
    )
    times = np.arange(num_frames) * frame_stride

    # Print a simple RTTM-like segment summary.
    print(f"Diarization for {args.audio}  ({num_frames} frames, {frame_stride * 1000:.0f} ms/frame)")
    for spk in range(num_speakers):
        segments = probs_to_segments(probs[:, spk] > args.threshold, frame_stride)
        if not segments:
            continue
        pretty = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in segments)
        print(f"  speaker {spk}: {pretty}")

    # Optional ground-truth overlay if a sibling <name>.json manifest exists (e.g. the test fixture).
    gt_path = os.path.splitext(args.audio)[0] + ".json"
    ground_truth = json.load(open(gt_path)) if os.path.exists(gt_path) else None

    # ---- plot ----
    colors = plt.get_cmap("tab10")
    fig, (ax_prob, ax_ribbon) = plt.subplots(
        2, 1, figsize=(11, 2 + 0.9 * num_speakers), sharex=True, height_ratios=[2, 1]
    )

    # Top panel: per-speaker probability curves with shaded active regions.
    for spk in range(num_speakers):
        ax_prob.plot(times, probs[:, spk], color=colors(spk), label=f"speaker {spk}")
        ax_prob.fill_between(
            times, 0, probs[:, spk], where=probs[:, spk] > args.threshold, color=colors(spk), alpha=0.2
        )
    ax_prob.axhline(args.threshold, ls="--", lw=0.8, color="gray")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("P(active)")
    ax_prob.set_title(f"Sortformer diarization — {os.path.basename(args.audio)}")
    ax_prob.legend(loc="upper right", ncol=num_speakers, fontsize=8)

    # Bottom panel: predicted activity ribbons (solid) and ground truth (hatched) per speaker.
    for spk in range(num_speakers):
        for a, b in probs_to_segments(probs[:, spk] > args.threshold, frame_stride):
            ax_ribbon.barh(spk, b - a, left=a, height=0.6, color=colors(spk))
    if ground_truth is not None:
        for entry in ground_truth.get("speakers", []):
            spk = entry["speaker"]
            ax_ribbon.barh(
                spk,
                entry["end_sec"] - entry["start_sec"],
                left=entry["start_sec"],
                height=0.85,
                fill=False,
                edgecolor="black",
                hatch="//",
                linewidth=0.8,
            )
        ax_ribbon.set_title("predicted (solid) vs ground truth (hatched)", fontsize=9)
    else:
        ax_ribbon.set_title("predicted speaker activity", fontsize=9)
    ax_ribbon.set_yticks(range(num_speakers))
    ax_ribbon.set_yticklabels([f"spk {s}" for s in range(num_speakers)])
    ax_ribbon.set_xlabel("time (s)")
    ax_ribbon.set_xlim(0, times[-1] if num_frames else 1)

    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
