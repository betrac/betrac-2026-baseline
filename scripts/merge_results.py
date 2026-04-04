#!/usr/bin/env python3
"""Merge result JSONL/JSON files from Slurm array jobs.

Supports both JSONL (one JSON object per line) and JSON array formats.

Usage:
    python scripts/merge_results.py outputs/slurm_array_*/*/omni_output.jsonl
    python scripts/merge_results.py outputs/slurm_array_*/*/omni_output.jsonl --output results/merged.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    """Load results from JSONL or JSON array file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    results = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Merge result JSONL/JSON files from array jobs"
    )
    parser.add_argument("files", nargs="+", help="JSONL or JSON result files to merge")
    parser.add_argument(
        "--output", default="results/merged_output.jsonl", help="Output JSONL path"
    )

    args = parser.parse_args()

    all_results = []
    seen_ids = set()
    files_loaded = 0

    for file_path in args.files:
        p = Path(file_path)
        if not p.exists():
            print(f"Warning: {p} not found, skipping", file=sys.stderr)
            continue
        results = load_results(str(p))
        for entry in results:
            entry_id = entry.get("id")
            if entry_id in seen_ids:
                print(
                    f"Warning: duplicate id '{entry_id}' in {p}, skipping",
                    file=sys.stderr,
                )
                continue
            seen_ids.add(entry_id)
            all_results.append(entry)
        files_loaded += 1

    if not all_results:
        print("No results found in specified files", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"Merged {len(all_results)} entries from {files_loaded} files -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
