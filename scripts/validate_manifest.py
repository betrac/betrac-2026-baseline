#!/usr/bin/env python3
"""Validate CSV manifest for correctness.

Checks that the manifest is well-formed and that referenced audio files exist.

Usage:
    python scripts/validate_manifest.py data/manifests/dataset.csv
    python scripts/validate_manifest.py data/manifests/dataset.csv --base-dir /path/to/project
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate CSV manifest")
    parser.add_argument("manifest", help="Path to CSV manifest file")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for resolving relative audio paths (default: manifest's parent)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    resolve_root = Path(args.base_dir) if args.base_dir else manifest_path.parent

    # Read manifest
    samples = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            print("Error: manifest missing required 'id' column", file=sys.stderr)
            return 1
        if "audio_path" not in (reader.fieldnames or []):
            print(
                "Error: manifest missing required 'audio_path' column", file=sys.stderr
            )
            return 1
        for row in reader:
            samples.append(row)

    print(f"Manifest: {manifest_path}")
    print(f"  Columns: {', '.join(reader.fieldnames or [])}")
    print(f"  Samples: {len(samples)}")

    if not samples:
        print("\nWarning: manifest is empty (header only)")
        return 0

    # Check audio files exist
    missing = 0
    ids = set()
    duplicates = 0
    for sample in samples:
        sample_id = sample["id"]
        if sample_id in ids:
            duplicates += 1
        ids.add(sample_id)

        audio_path = Path(sample["audio_path"])
        if not audio_path.is_absolute():
            audio_path = resolve_root / audio_path
        if not audio_path.exists():
            missing += 1
            if missing <= 5:
                print(f"  Missing: {sample_id} -> {audio_path}")

    if missing > 5:
        print(f"  ... and {missing - 5} more missing files")

    # Summary
    print(f"\n  Audio files found: {len(samples) - missing}/{len(samples)}")
    if duplicates:
        print(f"  Duplicate IDs: {duplicates}")

    if missing:
        print(f"\nError: {missing} audio file(s) not found")
        return 1

    if duplicates:
        print(f"\nWarning: {duplicates} duplicate ID(s) found")

    print("\nManifest is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
