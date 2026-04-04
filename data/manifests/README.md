# Manifest Format (Custom Data Only)

> **Note:** Manifests are only needed for custom audio data. The standard
> BeTraC dataset is loaded directly from HuggingFace using `--dataset`/`--split`.

## When to use manifests

Use a CSV manifest when you want to run the omni pipeline on your own
audio files instead of the BeTraC HuggingFace dataset:

```bash
# Custom data via manifest
make run-manifest MANIFEST=data/manifests/my_data.csv

# Standard BeTraC data (no manifest needed)
make run SPLIT=validation
```

## Required columns

| Column | Description |
|--------|-------------|
| `id` | Unique sample identifier (alphanumeric, hyphens, underscores) |
| `audio_path` | Path to audio file (absolute or relative to manifest directory) |

Supported audio formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`

## Example

```csv
id,audio_path
conv_001,audio/conv_001.opus
conv_002,audio/conv_002.opus
conv_003,audio/conv_003.wav
```

## Creating manifests

```bash
python scripts/create_manifest.py \
    --audio-dir /path/to/audio/ \
    --output data/manifests/my_data.csv
```

## Validation

```bash
python scripts/validate_manifest.py data/manifests/my_data.csv
```

Checks CSV format, verifies audio files exist, and reports statistics.
