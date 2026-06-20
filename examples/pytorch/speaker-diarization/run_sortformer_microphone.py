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
Real-time Sortformer speaker diarization from a live microphone.

Sortformer is a streaming diarization model: it keeps a bounded Arrival-Order Speaker Cache (AOSC) and processes audio
in fixed-size chunks, so memory and compute stay constant no matter how long you talk. This script captures audio from
your microphone, feeds it to the model chunk-by-chunk via `streaming_step`, prints, for every emitted chunk, which
of the (up to `num_speakers`) speakers are active, and shows a live scrolling plot of per-speaker activity
(disable with `--no-plot`).

A single microphone is the `batch_size=1` case, so the (ported) *synchronous* streaming path is exactly correct here.

Latency note: the model's default chunk is `chunk_len` (=188) subsampled frames, i.e. ~15 s, so by default it only
emits every ~15 s. Pass e.g. `--chunk-seconds 2` for a snappier, more interactive demo -- but note that shrinking the
chunk deviates from the configuration the model was validated with and may reduce accuracy.

Setup:

```bash
pip install sounddevice soundfile transformers torch
python examples/pytorch/speaker-diarization/run_sortformer_microphone.py --chunk-seconds 2
# list audio input devices, then pick one:
python examples/pytorch/speaker-diarization/run_sortformer_microphone.py --list-devices
python examples/pytorch/speaker-diarization/run_sortformer_microphone.py --input-device 3
```

Press Ctrl+C to stop.
"""

import argparse
import queue
import sys
import time

import numpy as np
import torch

from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Sortformer diarization from a microphone.")
    parser.add_argument("--model", default="cptspacemanspiff/diar_streaming_sortformer_4spk-v2", help="HF model id or path.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Override the chunk length (seconds). Smaller = lower latency but off the validated config. "
        "Defaults to the model's configured chunk length (~15 s).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for 'active'.")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index (see --list-devices).")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Auto-stop after this many seconds.")
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable the live activity plot (terminal output only)."
    )
    parser.add_argument("--plot-window", type=float, default=30.0, help="Seconds of history shown in the live plot.")
    return parser.parse_args()


class StreamingDiarizer:
    """Wraps a Sortformer model for incremental diarization of a growing audio stream.

    Feed it raw audio with `process(samples)`; it yields `(start_time, end_time, chunk_probs)` for each chunk that
    becomes ready. The mel features are (re)extracted over a small rolling audio buffer that is periodically trimmed,
    so memory stays bounded. The trim keeps a left-context + STFT-edge margin before the next read position, so emitted
    frames are unaffected by the rebase -- this is an approximation of NeMo's exact full-sequence feature extraction,
    accurate to within a couple of frames at the (discarded) chunk boundary.
    """

    # Frames discarded before the read position on each rebase, to stay clear of STFT center-padding edge effects.
    _EDGE_MARGIN_FRAMES = 4

    def __init__(self, model, feature_extractor, threshold: float = 0.5):
        self.model = model.eval()
        self.feature_extractor = feature_extractor
        self.threshold = threshold

        self.sampling_rate = feature_extractor.sampling_rate
        self.hop_length = feature_extractor.hop_length
        self.subsampling_factor = model.config.encoder_config.subsampling_factor
        self.frame_stride = self.subsampling_factor * self.hop_length / self.sampling_rate  # seconds per output frame

        self.chunk_mel = model.config.chunk_len * self.subsampling_factor
        self.left_mel = model.config.chunk_left_context * self.subsampling_factor
        self.right_mel = model.config.chunk_right_context * self.subsampling_factor

        self.state = model.init_streaming_state(batch_size=1)
        self._audio = np.zeros(0, dtype=np.float32)
        self._mel_consumed = 0  # mel frames consumed within the current (rolling) buffer
        self._out_frames = 0  # total output frames emitted (for absolute timestamps)

        # Diagnostics.
        self.chunk_seconds = model.config.chunk_len * self.frame_stride
        self.chunks_emitted = 0
        self.last_infer_ms = 0.0

    @torch.no_grad()
    def process(self, samples: np.ndarray):
        """Append new audio samples and yield any chunks that are now ready."""
        self._audio = np.concatenate([self._audio, samples.astype(np.float32)])
        # Need enough audio for the unconsumed mel frames plus one full chunk and its right look-ahead.
        min_samples = (self._mel_consumed + self.chunk_mel + self.right_mel + 2) * self.hop_length
        if len(self._audio) < min_samples:
            return

        features = self.feature_extractor(
            self._audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features  # (1, num_mel_frames, num_mel_bins); no peak-normalization for streaming
        num_frames = features.shape[1]

        while num_frames - self._mel_consumed >= self.chunk_mel + self.right_mel:
            left = min(self.left_mel, self._mel_consumed)
            start = self._mel_consumed
            window = features[:, start - left : start + self.chunk_mel + self.right_mel, :].to(self.model.device)
            t0 = time.perf_counter()
            chunk_probs, self.state = self.model.streaming_step(
                window, self.state, left_offset=left, right_offset=self.right_mel
            )
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
            chunk_probs = chunk_probs[0].float().cpu()  # (chunk_out_frames, num_speakers)
            start_time = self._out_frames * self.frame_stride
            self._out_frames += chunk_probs.shape[0]
            end_time = self._out_frames * self.frame_stride
            self._mel_consumed += self.chunk_mel
            self.chunks_emitted += 1
            yield start_time, end_time, chunk_probs

        # Trim/rebase the audio buffer to bound memory, keeping left-context + STFT-edge margin before the read head.
        keep_from = max(0, self._mel_consumed - self.left_mel - self._EDGE_MARGIN_FRAMES)
        if keep_from > 0:
            self._audio = self._audio[keep_from * self.hop_length :]
            self._mel_consumed -= keep_from

    def stats(self) -> dict:
        """Current internal state, for live diagnostics."""
        state = self.state
        return {
            "output_seconds": self._out_frames * self.frame_stride,
            "chunks_emitted": self.chunks_emitted,
            "last_infer_ms": self.last_infer_ms,
            "infer_ratio": (self.last_infer_ms / 1000.0) / self.chunk_seconds if self.chunk_seconds else 0.0,
            "spkcache_frames": state.spkcache.shape[1],
            "spkcache_cap": self.model.config.spkcache_len,
            "spkcache_compressed": state.spkcache_preds is not None,
            "fifo_frames": state.fifo.shape[1],
            "fifo_cap": self.model.config.fifo_len,
            "silence_frames": int(state.n_sil_frames.max().item()) if state.n_sil_frames is not None else 0,
        }


def format_chunk(start_time, end_time, probs, threshold):
    """Render one emitted chunk as a compact, human-readable line."""
    num_speakers = probs.shape[1]
    mean_prob = probs.mean(dim=0)
    active = [s for s in range(num_speakers) if (probs[:, s] > threshold).float().mean() > 0.1]
    header = f"[{start_time:6.1f}s - {end_time:6.1f}s]"
    if not active:
        return f"{header}  (silence)"
    parts = []
    for s in active:
        bars = int(round(float(mean_prob[s]) * 8))
        parts.append(f"S{s} {'█' * bars}{'·' * (8 - bars)} {float(mean_prob[s]):.2f}")
    overlap = " [OVERLAP]" if len(active) > 1 else ""
    return f"{header}  " + "   ".join(parts) + overlap


def format_status(mic_seconds, queue_blocks, stats):
    """One-line live diagnostics about the pipeline and model state."""
    lag = mic_seconds - stats["output_seconds"]
    cache = f"{stats['spkcache_frames']}/{stats['spkcache_cap']}"
    if stats["spkcache_compressed"]:
        cache += " (compressed)"
    return (
        f"mic {mic_seconds:6.1f}s | out {stats['output_seconds']:6.1f}s | lag {lag:5.1f}s | "
        f"queue {queue_blocks:2d} blk | chunks {stats['chunks_emitted']:3d} | "
        f"infer {stats['last_infer_ms']:4.0f}ms ({stats['infer_ratio']:.2f}x rt) | "
        f"cache {cache} | fifo {stats['fifo_frames']}/{stats['fifo_cap']} | sil {stats['silence_frames']}"
    )


class LivePlot:
    """A scrolling matplotlib plot of per-speaker P(active) over time, updated as chunks arrive."""

    def __init__(self, num_speakers, frame_stride, threshold, window_seconds=30.0):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.num_speakers = num_speakers
        self.frame_stride = frame_stride
        self.threshold = threshold
        self.window = window_seconds
        self.times = np.zeros(0, dtype=np.float64)
        self.probs = np.zeros((0, num_speakers), dtype=np.float32)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        cmap = plt.get_cmap("tab10")
        self.lines = [self.ax.plot([], [], color=cmap(s), label=f"speaker {s}")[0] for s in range(num_speakers)]
        self.ax.axhline(threshold, ls="--", lw=0.8, color="gray")
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("P(active)")
        self.ax.set_title("Sortformer live diarization")
        self.ax.legend(loc="upper left", ncol=num_speakers, fontsize=8)
        self.fig.subplots_adjust(bottom=0.18)
        # Live diagnostics line under the plot.
        self.status_text = self.fig.text(0.01, 0.02, "", fontsize=8, family="monospace")
        self.fig.show()

    def set_status(self, text):
        self.status_text.set_text(text)

    def add(self, start_time, probs):
        probs = probs.detach().cpu().numpy()
        times = start_time + np.arange(probs.shape[0]) * self.frame_stride
        self.times = np.concatenate([self.times, times])
        self.probs = np.concatenate([self.probs, probs], axis=0)
        # Keep only the most recent `window` seconds.
        keep = self.times >= self.times[-1] - self.window
        self.times, self.probs = self.times[keep], self.probs[keep]
        for s in range(self.num_speakers):
            self.lines[s].set_data(self.times, self.probs[:, s])
        self.ax.set_xlim(max(0.0, self.times[-1] - self.window), max(self.window, self.times[-1]))

    def refresh(self):
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def is_open(self):
        return self.plt.fignum_exists(self.fig.number)


def main():
    args = parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("This example needs `sounddevice`. Install it with: pip install sounddevice")

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Loading {args.model} ...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioFrameClassification.from_pretrained(args.model).eval()

    if args.chunk_seconds is not None:
        frame_stride = (
            model.config.encoder_config.subsampling_factor
            * feature_extractor.hop_length
            / feature_extractor.sampling_rate
        )
        chunk_len = max(1, round(args.chunk_seconds / frame_stride))
        model.config.chunk_len = chunk_len
        model.config.spkcache_update_period = chunk_len  # keep update period == chunk_len (fifo_len == 0)
        print(
            f"NOTE: chunk_len overridden to {chunk_len} frames (~{chunk_len * frame_stride:.1f}s). This deviates from "
            "the validated configuration and may reduce accuracy."
        )

    diarizer = StreamingDiarizer(model, feature_extractor, threshold=args.threshold)
    sampling_rate = feature_extractor.sampling_rate
    audio_queue: queue.Queue = queue.Queue()

    plot = None
    if not args.no_plot:
        try:
            plot = LivePlot(model.config.num_speakers, diarizer.frame_stride, args.threshold, args.plot_window)
        except Exception as error:  # no GUI backend / display -> fall back to terminal only
            print(f"Live plot disabled ({error}); continuing with terminal output only.")

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())  # mono

    print(
        f"Listening at {sampling_rate} Hz, emitting every ~{model.config.chunk_len * diarizer.frame_stride:.1f}s. "
        "Press Ctrl+C to stop.\n"
    )
    elapsed = 0.0  # seconds of audio captured from the mic
    last_status = 0.0
    last_refresh = 0.0
    try:
        with sd.InputStream(
            samplerate=sampling_rate, channels=1, dtype="float32", device=args.input_device, callback=audio_callback
        ):
            while True:
                # Drain ALL queued audio at once (one block, then everything else available). Pulling a single block
                # per iteration would let the queue grow without bound, since the GUI refresh caps the loop rate below
                # the callback's block rate -- which would make output lag real time more and more over time.
                blocks = []
                try:
                    blocks.append(audio_queue.get(timeout=0.1))  # block until there's something (or idle-poll)
                except queue.Empty:
                    pass
                while True:
                    try:
                        blocks.append(audio_queue.get_nowait())
                    except queue.Empty:
                        break

                if blocks:
                    samples = np.concatenate(blocks)
                    elapsed += len(samples) / sampling_rate
                    for start_time, end_time, probs in diarizer.process(samples):
                        print(format_chunk(start_time, end_time, probs, args.threshold), flush=True)
                        if plot is not None:
                            plot.add(start_time, probs)

                status = format_status(elapsed, audio_queue.qsize(), diarizer.stats())
                now = time.monotonic()
                if plot is not None:
                    plot.set_status(status)
                    if now - last_refresh >= 0.1:  # cap GUI redraw at ~10 Hz so it can't starve the audio drain
                        plot.refresh()
                        last_refresh = now
                    if not plot.is_open():  # user closed the window
                        break
                # Throttled terminal diagnostics (so --no-plot users see them too).
                if now - last_status >= 1.0:
                    print(f"  [{status}]", flush=True)
                    last_status = now
                if args.max_seconds is not None and elapsed >= args.max_seconds:
                    break
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
