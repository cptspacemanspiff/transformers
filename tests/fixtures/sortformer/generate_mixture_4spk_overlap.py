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
Regenerate the synthetic 4-speaker overlapping diarization fixture used by the Sortformer integration tests.

It mixes four distinct LibriSpeech (test-clean) speakers with staggered onsets and heavy mutual overlap, with
speaker 3 mixed at a lower level than the others to test robustness to level imbalance. It writes both the audio
file and a JSON manifest (ground-truth speaker segments + levels) next to this script:

    mixture_4spk_overlap.wav
    mixture_4spk_overlap.json

Usage:

```bash
python tests/fixtures/sortformer/generate_mixture_4spk_overlap.py
```

Requires `datasets`, `soundfile` and network access to stream LibriSpeech. The fixture is deterministic: the first
four distinct speakers encountered in the (ordered) `test-clean` stream are used, with fixed onsets and levels.
"""

import itertools
import json
import os

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


SR = 16000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WAV_PATH = os.path.join(OUT_DIR, "mixture_4spk_overlap.wav")
JSON_PATH = os.path.join(OUT_DIR, "mixture_4spk_overlap.json")

# Overlapping onsets (seconds) and per-speaker linear peak levels. Speaker 3 is deliberately quieter.
# Onsets are widely spaced so adjacent speakers only partially overlap (more space between them).
ONSETS = [0.0, 6.0, 12.0, 18.0]
LEVELS = [0.6, 0.6, 0.6, 0.3]
SPEECH_SECONDS = 8.0  # target speech length per speaker (utterances are concatenated to reach this)
MAX_STREAM_EXAMPLES = 2000  # how far to read the (ordered) stream while gathering per-speaker audio
FINAL_PEAK = 0.9  # final mix is peak-normalized to this to avoid clipping (preserves relative levels)


def main():
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True).cast_column(
        "audio", Audio(sampling_rate=SR)
    )

    target_samples = int(SPEECH_SECONDS * SR)
    chunks = {}  # speaker_id -> list of waveforms (concatenated later)
    for example in itertools.islice(ds, MAX_STREAM_EXAMPLES):
        speaker_id = str(example["speaker_id"])
        # Only gather for the first 4 distinct speakers we encounter.
        if speaker_id not in chunks and len(chunks) >= 4:
            continue
        bucket = chunks.setdefault(speaker_id, [])
        if sum(len(c) for c in bucket) < target_samples:
            bucket.append(example["audio"]["array"].astype(np.float32))
        if len(chunks) >= 4 and all(sum(len(c) for c in v) >= target_samples for v in chunks.values()):
            break
    if len(chunks) < 4:
        raise RuntimeError(f"Expected 4 distinct speakers, found {len(chunks)}")

    speaker_ids = list(chunks)
    # Concatenate each speaker's utterances and trim to the target speech length.
    picked = {sid: np.concatenate(chunks[sid])[:target_samples] for sid in speaker_ids}
    total = int(max(s + len(picked[sid]) / SR for s, sid in zip(ONSETS, speaker_ids)) * SR) + SR
    mix = np.zeros(total, dtype=np.float32)

    speakers = []
    for speaker, (sid, onset, level) in enumerate(zip(speaker_ids, ONSETS, LEVELS)):
        waveform = picked[sid]
        waveform = waveform / (np.abs(waveform).max() + 1e-8) * level
        start = int(round(onset * SR))
        mix[start : start + len(waveform)] += waveform
        speakers.append(
            {
                "speaker": speaker,
                "source_speaker_id": sid,
                "start_sec": round(onset, 3),
                "end_sec": round(onset + len(waveform) / SR, 3),
                "level": level,
            }
        )

    mix = mix / (np.abs(mix).max() + 1e-8) * FINAL_PEAK
    sf.write(WAV_PATH, mix, SR, subtype="PCM_16")

    # Verify the "at most 2 speakers overlap at once" invariant from the ground-truth segments.
    boundaries = sorted({t for spk in speakers for t in (spk["start_sec"], spk["end_sec"])})
    max_simultaneous = 0
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        mid = (a + b) / 2
        count = sum(1 for spk in speakers if spk["start_sec"] <= mid < spk["end_sec"])
        max_simultaneous = max(max_simultaneous, count)
    if max_simultaneous > 2:
        raise RuntimeError(f"Layout violates the max-2-overlap constraint: {max_simultaneous} speakers overlap")

    meta = {
        "description": (
            "Synthetic 4-speaker overlapping mixture for Sortformer diarization testing. Four distinct LibriSpeech "
            "speakers (~8s each) with widely-spaced onsets so that at most two speakers ever overlap at once, with "
            "single-speaker gaps in between; speaker 3 is mixed at a lower level than the others to test robustness "
            "to level imbalance."
        ),
        "source_dataset": "openslr/librispeech_asr (clean/test)",
        "source_license": "CC BY 4.0",
        "generator_script": os.path.basename(__file__),
        "audio_file": os.path.basename(WAV_PATH),
        "sampling_rate": SR,
        "num_speakers": 4,
        "max_simultaneous_speakers": max_simultaneous,
        "duration_sec": round(len(mix) / SR, 3),
        "frame_stride_sec": 0.08,
        "mix_final_peak": FINAL_PEAK,
        "speakers": speakers,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {WAV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(json.dumps(meta, indent=2))

    # Avoid the HF streaming-iterator cleanup hang on interpreter shutdown.
    os._exit(0)


if __name__ == "__main__":
    main()
