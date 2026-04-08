# BeTraC 2026 — End-to-End Omni Baseline

[![Challenge](https://img.shields.io/badge/SLT%202026-Challenge-blue.svg)](https://betrac.github.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

End-to-end audio-to-SOAP-note baseline for the [BeTraC 2026](https://betrac.github.io) challenge (IEEE SLT). Processes raw doctor–patient conversation audio directly into structured clinical SOAP notes using Qwen Omni models — no intermediate transcription.

```
HuggingFace dataset ──► [Omni step]  ──► output.jsonl
BeTraC/betrac-2026       steps/omni/      {id, summary, timing, ...}
(Opus audio)             single .venv
```

| Repository | Purpose |
|------------|---------|
| [betrac-2026-cascade](https://github.com/betrac/betrac-2026-cascade) | Cascade reference topline (ASR + LLM, not eligible) |
| [betrac-metrics](https://github.com/betrac/betrac-metrics) | Evaluation harness for SOAP note scoring |
| [BeTraC website](https://betrac.github.io) | Challenge description, rules, leaderboard |

---

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (`pip install uv` or `brew install uv`)
- The [BeTraC dataset](https://huggingface.co/datasets/BeTraC/betrac-2026) is public — no HuggingFace login required
- GPU recommended (CUDA or Apple Silicon MPS); CPU works but is slower

### Setup and test

```bash
git clone https://github.com/betrac/betrac-2026-baseline.git
cd betrac-2026-baseline
make setup          # creates steps/omni/.venv
make test           # smoke test with local sample audio (Qwen3-Omni-30B-A3B-Instruct)
make test-5         # 5 HuggingFace samples (Qwen2.5-Omni-3B)
```

Example outputs from these commands are included in `results/` for reference:
```bash
python3 scripts/show_results.py results/test_omni_output-example.jsonl
python3 scripts/show_results.py results/test_omni_5-example.jsonl
```

### Run inference

```bash
# Full validation set (default: Qwen3-Omni-30B-A3B-Instruct)
make run SPLIT=validation

# Lightweight 3B model, first 20 samples
make run SPLIT=validation MODEL=Qwen/Qwen2.5-Omni-3B LIMIT=20

# Custom audio data
make run-manifest MANIFEST=data/manifests/my_data.csv
```

### Output

Each sample produces a JSONL record:

```json
{
  "id": "sample_id",
  "summary": "S: Patient presents with...\nO: Vital signs...\nA: ...\nP: ...",
  "thinking": "",
  "omni_time_sec": 92.2,
  "total_time_sec": 92.2,
  "success": true,
  "error": ""
}
```

The `summary` field contains the SOAP note. The `thinking` field is populated for thinking models only.

---

## Verified Models

| Model | HuggingFace ID | Track | MPS | CUDA (A100) |
|-------|---------------|-------|-----|-------------|
| Qwen2.5-Omni-3B | `Qwen/Qwen2.5-Omni-3B` | Lightweight | ~70–120s | ~45s |
| Qwen2.5-Omni-7B | `Qwen/Qwen2.5-Omni-7B` | Heavyweight | ~160s (CPU) | ~90s |
| Qwen3-Omni-30B | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Heavyweight | ~260–360s | ~414s* |
| Qwen3-Omni-30B Thinking | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | Heavyweight | ~450–570s | ~1275s* |

MPS tested on Apple Silicon, CUDA on A100-PCIE-40GB (1 GPU + CPU offload for 30B models). Model family and device are auto-detected. *30B CUDA times are median across 400 validation samples — individual samples range from 3–50 min.

---

## SLURM Cluster Usage

**Generic scripts** (any SLURM cluster, manifest-based):

```bash
# Submit array job (auto-splits manifest across tasks)
bash scripts/submit_slurm.sh

# Custom model + cluster options
MODEL_ID=Qwen/Qwen2.5-Omni-3B MODEL_SHORT=qwen25-3b \
  bash scripts/submit_slurm.sh --account=MY_ACCOUNT --partition=gpu
```

Edit [scripts/run_omni.slurm](scripts/run_omni.slurm) to adjust `--gpus-per-task`, `--mem`, `--time` and uncomment `module load` lines for your cluster.

**Pre-configured experiment directories** (HuggingFace dataset mode):

The `experiments/` directory contains ready-to-run SLURM scripts for each model.
See [experiments/SLURM_SETUP.md](experiments/SLURM_SETUP.md) for adapting them
to your cluster.

```bash
cd experiments/Exp0001-qwen25-3b/
TOTAL=400 bash slurm/submit_omni.sh   # submits array + combine jobs
```

**Local run** (no SLURM):

```bash
bash scripts/run_local.sh             # generic
bash experiments/Exp0001-qwen25-3b/run_local.sh  # experiment-specific
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues (HF rate limits,
GPU memory, NFS errors).

---

## Customization

This baseline is designed to be modified:

- **Swap models:** Change `--model` to any HuggingFace omni model. Family is auto-detected.
- **Custom prompts:** Edit `conf/prompt/default.yaml` or pass `--prompt path/to/prompt.yaml`
- **Custom data:** Create a CSV manifest with `scripts/create_manifest.py`, run with `--manifest`
- **Model config:** Edit `MODEL_CONFIGS` in `run_omni.py` to add per-model defaults
- **Resume interrupted runs:** Re-run the same command with `--resume` to skip completed samples

---

## Reference

<details>
<summary><b>CLI Options</b></summary>

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `BeTraC/betrac-2026` | HuggingFace dataset name |
| `--split` | `validation` | Dataset split to process |
| `--limit` | (all) | Process only the first N samples |
| `--manifest` | (none) | CSV manifest file (overrides `--dataset`/`--split`) |
| `--output` | (required) | Output JSONL file path |
| `--model` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Model name or local path |
| `--device` | `auto` | Device: `auto`, `cpu`, `mps`, or CUDA ID |
| `--prompt` | built-in SOAP | Path to custom prompt YAML (`text:` key) |
| `--resume` | off | Skip samples already in the output file |
| `--thinking` | auto-detected | Force thinking-model generation mode |
| `--use-vllm` | off | Use vLLM backend (faster on multi-GPU) |
| `--use-flash-attn2` | off | Enable Flash Attention 2 (A100+) |
| `--seed` | 42 | Random seed |

</details>

<details>
<summary><b>Makefile Targets</b></summary>

Run `make help` to see all targets:

| Target | Description |
|--------|-------------|
| `make setup` | Create virtual environment and install dependencies |
| `make run` | Run on HuggingFace dataset (primary) |
| `make run-manifest` | Run on custom CSV manifest (fallback) |
| `make test` | Smoke test with local example audio |
| `make test-5` | Smoke test on 5 HuggingFace samples (3B model) |
| `make test-hf` | Run on HuggingFace samples (configurable MODEL, LIMIT) |
| `make lint` | Run ruff linter and mypy |
| `make lint-fix` | Auto-fix linting issues |
| `make format` | Format code with ruff |
| `make clean` | Remove venv and results |

</details>

<details>
<summary><b>Project Structure</b></summary>

```
betrac-2026-baseline/
├── steps/omni/
│   ├── run_omni.py          # Main inference script
│   └── pyproject.toml       # Pinned dependencies
├── conf/prompt/
│   └── default.yaml         # SOAP note prompt template
├── data/
│   ├── manifests/           # CSV manifests for custom data
│   └── audio/               # Sample audio for smoke tests
├── scripts/
│   ├── submit_slurm.sh      # SLURM array job submission
│   ├── run_omni.slurm       # SLURM job template
│   ├── run_combine.slurm    # SLURM combine template
│   ├── run_local.sh         # Local (non-SLURM) runner
│   ├── create_manifest.py   # Generate manifests from audio dirs
│   ├── validate_manifest.py # Validate manifest format
│   ├── merge_results.py     # Merge JSONL from parallel jobs
│   └── show_results.py      # Pretty-print results
├── tests/                   # Test suite
├── results/                 # Output (includes *-example.jsonl for reference)
├── examples/                # Example SOAP note and JSONL format
├── Makefile                 # Build and run automation
└── pyproject.toml           # Project metadata
```

</details>

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The audio processing code in [run_omni.py](steps/omni/run_omni.py) is derived from Alibaba Cloud's Qwen Omni example cookbooks (Apache 2.0, Copyright 2025 Alibaba Cloud).

## Acknowledgments

The [Synth-DoPaCo](https://huggingface.co/datasets/BeTraC/betrac-2026) dataset
used in BeTraC 2026 was created by the Play-Your-Part team during the
[JSALT 2025](https://www.clsp.jhu.edu/workshops/) workshop at Johns Hopkins
University hosted at Brno University.
