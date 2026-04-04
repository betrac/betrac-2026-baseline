#!/usr/bin/env python3
"""Create CSV manifest from directory of audio files.

Scans a directory for audio files and creates a CSV manifest compatible
with the omni pipeline (steps/omni/run_omni.py).

Usage:
    python scripts/create_manifest.py --audio-dir data/audio/ --output data/manifests/dataset.csv
"""

import argparse
import csv
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wma"}


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in a directory (recursively)."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create CSV manifest from audio directory"
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing audio files (.wav, .mp3, .flac, .m4a, etc.)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Store audio paths relative to the current directory (default: absolute)",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a directory", file=sys.stderr)
        return 1

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"Error: no audio files found in {audio_dir}", file=sys.stderr)
        return 1

    rows = []
    for f in audio_files:
        audio_path = str(f) if args.relative else str(f.resolve())
        rows.append({"id": f.stem, "audio_path": audio_path})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["id", "audio_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {args.output} with {len(rows)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
