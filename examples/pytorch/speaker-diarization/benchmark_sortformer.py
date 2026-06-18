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
Benchmark Sortformer inference latency vs. window width.

This measures the part of the model that is recomputed on every streaming step: the Conformer encoder layers +
Transformer encoder + speaker head, run over a window of already-subsampled encoder frames. The convolutional
subsampling (pre-encode) is intentionally excluded -- it runs once per chunk on the new audio only, whereas the
back-end re-attends the whole `[cache + chunk]` window every step.

Each encoder frame spans 80 ms, so a window of W frames corresponds to W * 0.08 s of audio.

Optionally applies post-training quantization (`--quantize`) and/or `torch.compile` (`--compile`, dynamic-shape so it
does not recompile per width). When quantizing, also reports `max|Δ|` of the output vs. the fp32 model on the same
(random) input -- a numerical sanity signal, NOT a real accuracy measurement (use real audio for that).

Example:

```bash
python examples/pytorch/speaker-diarization/benchmark_sortformer.py \
    --model nvidia/diar_streaming_sortformer_4spk-v2-hf --widths 1 10 100 188 200 400 800 \
    --quantize torchao --compile
```
"""

import argparse
import statistics
import time

import torch

from transformers import AutoModelForAudioFrameClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the Sortformer back-end latency vs. window width.")
    parser.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2-hf", help="HF model id or path.")
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 188, 200, 400, 800],
        help="Window widths to benchmark, in 80 ms encoder frames (e.g. a [cache + chunk] window of W frames).",
    )
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations per width.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per width (not timed).")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU thread count (default: leave as-is).")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu).")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Compute dtype."
    )
    parser.add_argument(
        "--quantize",
        default="none",
        choices=["none", "dynamic", "torchao"],
        help="PTQ: 'dynamic' = torch dynamic int8 on Linear; 'torchao' = int8 dynamic-act + int8-weight.",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the back-end (dynamic shapes).")
    parser.add_argument(
        "--compile-mode",
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (only used with --compile).",
    )
    return parser.parse_args()


def run_backend(model, embeddings, lengths):
    """Run the recomputed-every-step part: Conformer layers -> proj -> Transformer -> speaker head (sigmoid)."""
    encoder_embeddings, _ = model._run_conformer_on_embeddings(embeddings, lengths)
    return model._forward_infer(encoder_embeddings, lengths)


def apply_quantization(model, method):
    if method == "dynamic":
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    if method == "torchao":
        try:
            from torchao.quantization import quantize_

            try:  # torchao >= ~0.8 config-class API
                from torchao.quantization import Int8DynamicActivationInt8WeightConfig

                config = Int8DynamicActivationInt8WeightConfig()
            except ImportError:  # older function API
                from torchao.quantization import int8_dynamic_activation_int8_weight

                config = int8_dynamic_activation_int8_weight()
        except ImportError as error:
            raise SystemExit(f"--quantize torchao needs torchao installed: {error}")
        quantize_(model, config)
        return model
    return model


def main():
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)

    model = AutoModelForAudioFrameClassification.from_pretrained(args.model, dtype=dtype).to(device).eval()
    fc_d_model = model.config.encoder_config.hidden_size  # dim of subsampled encoder frames (pre-encode embeddings)

    # Fixed random inputs per width (reused for fp32 reference + quantized run, so the accuracy diff is meaningful).
    torch.manual_seed(0)
    inputs = {
        width: (
            torch.randn(1, width, fc_d_model, device=device, dtype=dtype),
            torch.tensor([width], device=device),
        )
        for width in args.widths
    }

    # fp32 reference outputs (before quantization) for the accuracy column.
    references = {}
    if args.quantize != "none":
        with torch.inference_mode():
            for width, (embeddings, lengths) in inputs.items():
                references[width] = run_backend(model, embeddings, lengths).clone()

    model = apply_quantization(model, args.quantize)

    def backend_call(embeddings, lengths):
        return run_backend(model, embeddings, lengths)

    if args.compile:
        backend_call = torch.compile(backend_call, mode=args.compile_mode, dynamic=True)

    print(f"model={args.model}  device={device}  dtype={args.dtype}  threads={torch.get_num_threads()}")
    print(f"quantize={args.quantize}  compile={args.compile}" + (" (dynamic)" if args.compile else ""))
    header = f"{'width':>6} {'mean(ms)':>9} {'min(ms)':>8} {'max(ms)':>8} {'infer/s':>9}"
    if args.compile:
        header += f" {'compile(ms)':>11}"
    if references:
        header += f" {'max|Δ|':>8}"
    print(header)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for width in args.widths:
        embeddings, lengths = inputs[width]
        with torch.inference_mode():
            # Time the first (cold) call separately: with --compile this is the compilation cost for this shape.
            synchronize()
            start = time.perf_counter()
            out = backend_call(embeddings, lengths)
            synchronize()
            compile_s = time.perf_counter() - start

            for _ in range(max(0, args.warmup - 1)):
                out = backend_call(embeddings, lengths)
            synchronize()

            timings = []
            for _ in range(args.iters):
                synchronize()
                start = time.perf_counter()
                out = backend_call(embeddings, lengths)
                synchronize()
                timings.append(time.perf_counter() - start)

        mean_s = statistics.mean(timings)
        row = (
            f"{width:>6} {mean_s * 1000:>9.1f} {min(timings) * 1000:>8.1f} {max(timings) * 1000:>8.1f} "
            f"{1 / mean_s:>9.1f}"
        )
        if args.compile:
            row += f" {compile_s * 1000:>11.0f}"
        if references:
            row += f" {(out.float() - references[width].float()).abs().max().item():>8.4f}"
        print(row)


if __name__ == "__main__":
    main()
