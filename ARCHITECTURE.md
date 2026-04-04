# Architecture

System architecture for the BeTraC 2026 end-to-end omni baseline.

## Overview

The baseline processes raw doctor–patient audio directly into structured SOAP notes using a single Qwen Omni model — no intermediate transcription step. It supports both the Qwen2.5-Omni (dense) and Qwen3-Omni (MoE) model families.

## Pipeline

```
Data source                 Inference                    Output
───────────                 ─────────                    ──────
HuggingFace dataset    ┐
  BeTraC/betrac-2026   ├──► run_omni.py ──► JSONL file
  (Opus audio)         │    steps/omni/      {id, summary, thinking, timing, ...}
                       │
CSV manifest (custom)  ┘
```

## Processing Model

Each audio sample is processed in a **separate subprocess** (via `multiprocessing.spawn`). This provides:

- **GPU memory isolation** — model memory is released after each sample
- **Fault tolerance** — a crash in one sample doesn't kill the entire run
- **Resume support** — `--resume` skips already-processed sample IDs

The subprocess lifecycle per sample:
1. Load model and processor (`_load_model_processor`)
2. Build chat messages with audio input
3. Run inference (`run_model`)
4. Return result via multiprocessing queue
5. Subprocess exits, freeing all memory

## Model Config Registry

`MODEL_CONFIGS` in `run_omni.py` provides per-model defaults:

| Model | Family | max_new_tokens | Device notes |
|-------|--------|---------------|--------------|
| Qwen2.5-Omni-3B | qwen2.5-omni | 4096 | CUDA/MPS/CPU |
| Qwen2.5-Omni-7B | qwen2.5-omni | 4096 | CUDA/CPU (MPS auto-overridden to CPU) |
| Qwen3-Omni-30B-Instruct | qwen3-omni-moe | 8192 | CUDA/MPS (~3B active params) |
| Qwen3-Omni-30B-Thinking | qwen3-omni-moe | 32768 | CUDA/MPS (~3B active params) |

The model family is auto-detected from the model name. CLI flags (`--device`, `--model`) override config defaults.

## Key Components

| File | Purpose |
|------|---------|
| `steps/omni/run_omni.py` | Main inference script — model loading, generation, output |
| `steps/omni/pyproject.toml` | Dependencies (torch, transformers, etc.) |
| `conf/prompt/default.yaml` | SOAP note generation prompt |
| `scripts/submit_slurm.sh` | SLURM array job orchestration |
| `scripts/run_local.sh` | Local (non-SLURM) runner |

## Data Flow

### HuggingFace path (default)
1. `load_hf_samples()` streams from `BeTraC/betrac-2026` WebDataset
2. Extracts `.opus` audio bytes, `.soap.txt` reference, `.json` metadata
3. `write_audio_tempfile()` writes audio to a temp `.opus` file
4. Subprocess processes the temp file, temp file is deleted

### Manifest path (custom data)
1. `read_manifest()` reads CSV with `id` and `audio_path` columns
2. Validates all audio files exist before starting
3. Subprocess processes each audio file directly

## SLURM Parallel Processing

For large-scale runs, `submit_slurm.sh` orchestrates:

1. **Array job** (`run_omni.slurm`) — each task processes a slice of the manifest
   - Manifest is sliced by line number: task *i* gets lines `[i*N+2, (i+1)*N+1]`
   - Each task writes `summaries_task_<i>.jsonl`
2. **Combine job** (`run_combine.slurm`) — runs after all array tasks succeed
   - Concatenates per-task JSONL files into `summaries.jsonl`
   - Triggered via `--dependency=afterok:<array_job_id>`

## Output Schema

```json
{
  "id": "sample_uuid",
  "summary": "S: ...\nO: ...\nA: ...\nP: ...",
  "thinking": "(chain-of-thought, thinking models only)",
  "omni_time_sec": 92.2,
  "total_time_sec": 92.2,
  "success": true,
  "error": ""
}
```
