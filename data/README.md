# Data Directory

## Structure

- `audio/` — Sample audio files for smoke tests
- `manifests/` — CSV manifests for custom data (see [manifests/README.md](manifests/README.md))

## Standard Data

The BeTraC dataset loads directly from HuggingFace — no local data files needed:

```bash
make run SPLIT=validation
```

## Custom Data

For your own audio files, create a CSV manifest and run:

```bash
python scripts/create_manifest.py --audio-dir /path/to/audio/ --output data/manifests/my_data.csv
make run-manifest MANIFEST=data/manifests/my_data.csv
```
