#!/usr/bin/env python3
"""Pretty-print results from JSONL or JSON output files.

Supports both JSONL (one JSON object per line) and JSON array formats.

Usage:
    python scripts/show_results.py results/omni_output.jsonl
    python scripts/show_results.py results/omni_output.jsonl --brief
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

    # Try JSON array first
    if text.startswith("["):
        return json.loads(text)

    # Otherwise treat as JSONL
    results = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print pipeline results")
    parser.add_argument("input", help="Path to JSONL or JSON results file")
    parser.add_argument("--brief", action="store_true", help="Show only ID, success, and timing")
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Print only the summary/notes text for each entry (no headers or timing)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    results = load_results(args.input)
    if not results:
        print("No results found.")
        return

    print(f"Results: {len(results)} entries from {args.input}\n")

    for i, entry in enumerate(results):
        sample_id = entry.get("id", "unknown")
        success = entry.get("success", False)
        status = "OK" if success else "FAILED"

        # Gather timing info
        timing_parts = []
        for key in ("omni_time_sec", "asr_time_sec", "llm_time_sec", "total_time_sec"):
            if key in entry:
                label = key.replace("_time_sec", "").upper()
                timing_parts.append(f"{label}: {entry[key]:.1f}s")

        timing_str = ", ".join(timing_parts) if timing_parts else "N/A"

        print(f"--- [{i + 1}] {sample_id} ({status}) [{timing_str}] ---")

        if args.notes_only:
            if entry.get("summary"):
                print(f"{entry['summary']}")
        elif not args.brief:
            if entry.get("error"):
                print(f"  Error: {entry['error']}")
            if entry.get("transcript"):
                transcript = entry["transcript"]
                if len(transcript) > 200:
                    transcript = transcript[:200] + "..."
                print(f"  Transcript: {transcript}")
            if entry.get("summary"):
                print(f"  Summary:\n{entry['summary']}\n")
        print()


if __name__ == "__main__":
    main()
