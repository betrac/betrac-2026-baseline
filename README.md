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
| [experiments/RESULTS.md](experiments/RESULTS.md) | Validation-set baseline scores and the audio-truncation impact study |

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

### Eval set

The eval dataset (`BeTraC/betrac-2026-eval`) will be released on **July 24, 2026**
to registered teams. Once your team has been granted access, export your
HuggingFace token and run:

```bash
export HF_TOKEN=hf_your_token_here   # or add to ~/.bashrc

# Local
make run DATASET=BeTraC/betrac-2026-eval SPLIT=eval MODEL=Qwen/Qwen2.5-Omni-3B

# SLURM cluster (see experiments/Exp0001-qwen25-3b/ and experiments/SLURM_SETUP.md)
cd experiments/Exp0001-qwen25-3b
DATASET=BeTraC/betrac-2026-eval SPLIT=eval TOTAL=875 SAMPLES_PER_TASK=20 \
  bash slurm/submit_omni.sh
```

`TOTAL=875` skips the slow sample auto-count; omit it to auto-detect.
The submit script pre-caches the dataset and model weights on the login node
before submitting jobs. Set `SKIP_CACHE=1` to skip if already cached.

### Output

Each sample produces a JSONL record:

```json
{
  "id": "sample_id",
  "summary": "S: Patient presents with...\nO: Vital signs...\nA: ...\nP: ...",
  "thinking": "",
  "omni_time_sec": 92.2,
  "total_time_sec": 92.2,
  "audio_sec": 530.4,
  "audio_used_sec": 530.4,
  "success": true,
  "error": ""
}
```

The `summary` field contains the SOAP note. The `thinking` field is populated for
thinking models only. `audio_sec` is the recording length and `audio_used_sec` is
how much of it was fed to the model — if these differ, the tail was dropped to fit
the model context (see [Audio length](#audio-length)).

---

## Verified Models

| Model | HuggingFace ID | Track | MPS† | A100 40 GB | H100 93 GB |
|-------|---------------|-------|------|-----------|-----------|
| Qwen2.5-Omni-3B | `Qwen/Qwen2.5-Omni-3B` | Lightweight | ~70–120s† | ~42s | — |
| Qwen2.5-Omni-7B | `Qwen/Qwen2.5-Omni-7B` | Heavyweight | ~160s (CPU)† | ~37s | — |
| Qwen3-Omni-30B | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Heavyweight | ~260–360s† | ~626s* | ~140–190s |
| Qwen3-Omni-30B Thinking | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | Heavyweight | ~450–570s† | ~1133s* | ~270–380s |

A100/H100 times are per-sample medians over the 400 **full-length** validation
dialogs (mean 8m 50s of audio) after the audio-truncation fix; the 30B A100
configuration is 1 GPU + CPU offload, the H100 runs fully on-card. *Individual
30B A100 samples range from ~3 to ~67 min; the longest ~2 % of dialogs trigger
the OOM retry ladder there. †MPS timings were measured on Apple Silicon
**before** the fix (audio cut at 300 s / 30 s) and have not been re-measured;
with full audio expect roughly 1.5–2× these figures. Model family and device
are auto-detected.

---

## Audio length

BeTraC consultations are long — the validation split averages **8m 50s** and runs
up to **24m 34s**. The Omni processors do **not** handle that out of the box.

Both families frame audio with a `WhisperFeatureExtractor`, whose `__call__`
defaults to `truncation=True` with `max_length = chunk_length * sampling_rate`.
`chunk_length` comes from each checkpoint's `preprocessor_config.json`, and the
cut happens silently — no warning, no error, just a shorter tensor:

| Checkpoint | `chunk_length` | Audio the model actually hears |
|------------|----------------|--------------------------------|
| `Qwen2.5-Omni-3B` / `-7B` | `300` | first **300 s** |
| `Qwen3-Omni-30B-A3B-*` | *absent* → Whisper default `30` | first **30 s** |

`run_omni.py` disables that by passing `truncation=False` to the processor. If you
build your own pipeline on top of `transformers`, **you need this line too**:

```python
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True,
    truncation=False,            # <- without this the audio is cut short
    use_audio_in_video=True,
)
```

Verify it on any checkpoint — this downloads processor files only, no weights:

```bash
python scripts/check_audio_truncation.py
python scripts/check_audio_truncation.py --models Qwen/Qwen2.5-Omni-7B --duration 1200
```

### GPU memory, and why the default is slow

The context window is not the only limit. `run_omni.py` defaults to **one GPU
plus CPU offload** (`--num-gpus 1`), which on a 40 GB card runs the 30B models
out of memory around **~1000 s of audio** — well before the 4372 s context cap.
Long recordings hit that wall.

`--num-gpus N` shards the model across N GPUs instead, reserving headroom on
each so the KV cache still fits. Measured on ~1450 s consultations:

| Configuration | Audio processed | Time | Reliable? |
|---------------|-----------------|------|-----------|
| 1× A100 40 GB + CPU offload (default) | 870 s — OOM, auto-shortened | 954 s | yes, but truncates |
| 2× A100 40 GB, `--num-gpus 2` | 1450 s (full) | 257 s | **no — output corrupted** |
| **1× H100 93 GB** | **1474 s (full)** | **304 s** | **yes** |

**A single GPU with enough memory to hold the model is the reliable
configuration.** A 30B MoE in bf16 needs roughly 60 GB, so a 80–96 GB card runs
it fully resident with no offload and no sharding.

Two 40 GB cards are *marginal* for the 30B and we recommend **against** them:
with `max_memory` reserving 10 GiB per card, only ~58 GiB remains for ~60 GB of
weights, so a few layers still land on CPU. In testing this configuration went
**0 for 3**: two dialogs hung past a 2-hour walltime, and the one that
"completed" (1450 s of audio in 257 s) emitted ~90 characters of degenerate
noise with `success: true` — silent output corruption, the worst failure mode.
`run_omni.py` now rejects near-empty summaries as failures, but treat
`--num-gpus` as experimental: verify a sharded output against a single-GPU
output on the same sample before trusting it.

If a sample runs out of memory it is retried automatically with progressively
shorter audio (60 %, then 35 %, floored at 300 s) rather than lost, and
`audio_used_sec` records what was actually processed.

A note on cost: roughly **half** of each 30B sample is model loading, because
every sample is processed in a fresh subprocess for memory isolation. On an
H100, 162 s of a ~300 s sample is `from_pretrained`, and inference itself is
nearly constant (~140 s) whether the audio is 344 s or 1474 s.

### How long can the audio be?

The audio tower is windowed and handles arbitrary lengths, so the real bound is
the thinker LLM context. Audio costs a fixed number of tokens per second:

| Family | Audio tokens/s | Context | Default `--max-audio-seconds` |
|--------|----------------|---------|-------------------------------|
| Qwen2.5-Omni | 25.0 | 32 768 | **1126 s** (18m 46s) |
| Qwen3-Omni-MoE Instruct | 13.0 | 65 536 | **4372 s** (72m 52s) |
| Qwen3-Omni-MoE Thinking | 13.0 | 65 536 | **3742 s** (62m 22s) |

`run_omni.py` derives this cap from `MODEL_CONFIGS` and applies it at audio-load
time, logging a warning whenever a recording is longer. Override it with
`--max-audio-seconds` (CLI) or `MAX_AUDIO_SECONDS` (every runner script); pass a
negative value to disable the cap entirely.

On the validation split the default cap affects **8 of 400** dialogs for the
Qwen2.5 models (99.2 % of all audio still reaches the model) and **none** for the
Qwen3 models.

The cap is derived assuming the built-in SOAP prompt (~170 tokens), which leaves
roughly 350 tokens of headroom at full length. A substantially longer custom
`--prompt` eats into that; `run_omni.py` logs a warning if the prompt plus the
generation budget would overrun the context, and you can lower
`--max-audio-seconds` to buy the room back.

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
| `--max-audio-seconds` | from model context | Longest audio fed to the model (~1126s Qwen2.5-Omni, ~4372s Qwen3-Instruct, ~3742s Qwen3-Thinking). Negative = no cap |
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
