#!/usr/bin/env python3
"""Diagnose (and verify the fix for) silent audio truncation in Omni processors.

Both Qwen Omni processor families frame audio with a `WhisperFeatureExtractor`
whose ``__call__`` defaults to ``truncation=True`` with
``max_length = chunk_length * sampling_rate``. ``chunk_length`` comes from each
checkpoint's ``preprocessor_config.json``:

    Qwen2.5-Omni     chunk_length=300   -> audio silently cut at 300 s
    Qwen3-Omni-MoE   chunk_length unset -> Whisper default 30 -> cut at 30 s

This script feeds a synthetic long recording through the real processor and
reports how much audio actually survives, with and without the fix. It only
downloads processor files (a few MB) — model weights are never loaded.

Usage:
    python scripts/check_audio_truncation.py
    python scripts/check_audio_truncation.py --models Qwen/Qwen2.5-Omni-3B
    python scripts/check_audio_truncation.py --duration 1200
"""
# Copyright 2026 BeTraC contributors
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

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen2.5-Omni-7B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking",
]

SAMPLE_RATE = 16000


def kept_seconds(inputs, feature_extractor) -> float:
    """Seconds of audio that survived the feature extractor."""
    frames = int(inputs["feature_attention_mask"].sum())
    return frames * feature_extractor.hop_length / feature_extractor.sampling_rate


def check_model(model_id: str, duration: float) -> bool:
    """Report truncation behaviour for one model. Returns True if the fix works."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    fe = processor.feature_extractor

    audio = (np.random.randn(int(SAMPLE_RATE * duration)) * 0.01).astype(np.float32)
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": "dummy.wav"},
            {"type": "text", "text": "Summarize this consultation as a SOAP note."},
        ],
    }]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    def run(**kwargs):
        return processor(
            text=text, audio=[audio], return_tensors="pt",
            padding=True, use_audio_in_video=False, **kwargs,
        )

    buggy = run()
    fixed = run(truncation=False)

    kept_buggy = kept_seconds(buggy, fe)
    kept_fixed = kept_seconds(fixed, fe)
    tokens_buggy = int(buggy["input_ids"].shape[1])
    tokens_fixed = int(fixed["input_ids"].shape[1])
    rate = (tokens_fixed - tokens_buggy) / max(kept_fixed - kept_buggy, 1e-9)

    print(f"\n{model_id}")
    print(f"  feature extractor : chunk_length={fe.chunk_length} "
          f"-> hard cap {fe.n_samples / fe.sampling_rate:.0f}s")
    print(f"  input audio       : {duration:.0f}s")
    print(f"  default kwargs    : {kept_buggy:7.1f}s reaches the model "
          f"({100 * kept_buggy / duration:5.1f}%), {tokens_buggy} prompt tokens")
    print(f"  truncation=False  : {kept_fixed:7.1f}s reaches the model "
          f"({100 * kept_fixed / duration:5.1f}%), {tokens_fixed} prompt tokens")
    print(f"  measured rate     : {rate:.1f} audio tokens/second")

    ok = kept_fixed >= duration - 1.0
    if kept_buggy < duration - 1.0:
        print(f"  => TRUNCATION CONFIRMED at {kept_buggy:.0f}s with default kwargs")
    else:
        print("  => no truncation with default kwargs")
    print(f"  => fix {'WORKS' if ok else 'DID NOT WORK'}")
    return ok


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model IDs to check (default: all four baselines)")
    parser.add_argument("--duration", type=float, default=900.0,
                        help="Synthetic audio length in seconds (default: %(default)s)")
    args = parser.parse_args()

    all_ok = True
    for model_id in args.models:
        try:
            all_ok &= check_model(model_id, args.duration)
        except Exception as exc:  # pragma: no cover - network/cache issues
            print(f"\n{model_id}\n  ERROR: {type(exc).__name__}: {exc}")
            all_ok = False

    print("\n" + "=" * 70)
    print("All processors pass full-length audio with truncation=False."
          if all_ok else "One or more checks failed — see above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
