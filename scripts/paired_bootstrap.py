#!/usr/bin/env python3
"""Paired bootstrap significance test for before/after per-dialog scores.

Used for every significance claim in experiments/RESULTS.md. Takes two
`concept_per_dialog.jsonl` files as produced by `btc-eval evaluate --output`,
pairs them by dialog id, and bootstraps the mean per-dialog delta.

Usage:
    python scripts/paired_bootstrap.py before/concept_per_dialog.jsonl \
                                       after/concept_per_dialog.jsonl
    python scripts/paired_bootstrap.py before.jsonl after.jsonl --metric recall
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
import json
import sys

import numpy as np


def load(path: str, metric: str) -> dict[str, float]:
    """Read {id: metric} from a per-dialog JSONL."""
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = float(r[metric])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("before", help="per-dialog JSONL for the baseline run")
    ap.add_argument("after", help="per-dialog JSONL for the treatment run")
    ap.add_argument("--metric", default="f1",
                    help="field to compare (default: %(default)s)")
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    before = load(args.before, args.metric)
    after = load(args.after, args.metric)
    ids = sorted(set(before) & set(after))
    if len(ids) < len(before) or len(ids) < len(after):
        print(f"warning: pairing dropped "
              f"{max(len(before), len(after)) - len(ids)} unmatched dialogs",
              file=sys.stderr)

    delta = np.array([after[i] - before[i] for i in ids])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(delta), size=(args.resamples, len(delta)))
    means = delta[idx].mean(axis=1)

    lo, hi = np.percentile(means, [2.5, 97.5])
    n_le0 = int((means <= 0).sum())
    print(f"paired dialogs        : {len(ids)}")
    print(f"mean per-dialog delta : {delta.mean():+.5f}")
    print(f"95% bootstrap CI      : [{lo:+.5f}, {hi:+.5f}]")
    print(f"resamples with <= 0   : {n_le0} / {args.resamples}"
          f"  (p < {max(n_le0, 1) / args.resamples:.2e}"
          f"{' — reported as an upper bound' if n_le0 == 0 else ''})")
    print(f"dialogs improved      : {(delta > 0).sum()} / {len(delta)}"
          f"   worsened: {(delta < 0).sum()}   unchanged: {(delta == 0).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
