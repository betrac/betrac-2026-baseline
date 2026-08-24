# Exp0002 — Qwen2.5-Omni-7B

Heavyweight-track baseline using the medium-sized dense Qwen Omni model.

| Field | Value |
|-------|-------|
| Model ID | `Qwen/Qwen2.5-Omni-7B` |
| Parameters | 7B (dense) |
| Family | Qwen2.5-Omni |
| Track | Heavyweight |
| GPUs | 1 |
| Memory | 80 GB |
| Walltime/task | 1 hour |

## Audio length

This checkpoint's `preprocessor_config.json` made the `WhisperFeatureExtractor`
silently truncate audio at **300 s** (Qwen2.5-Omni family). `run_omni.py` now passes
`truncation=False`, so the whole recording reaches the model, bounded instead by
`--max-audio-seconds` — **1126 s (18m 46s)** by default here, derived from the thinker
context window.

Pre-fix results are kept for comparison under `results-truncated-bug/`.
See the repository [README](../../README.md#audio-length) and
`scripts/check_audio_truncation.py`.

```bash
# Shorter cap (faster, less context)
TOTAL=400 MAX_AUDIO_SECONDS=600 bash slurm/submit_omni.sh
```

## Quick start

```bash
cd experiments/Exp0002-qwen25-7b/

# SLURM — validation set (400 samples)
TOTAL=400 bash slurm/submit_omni.sh

# SLURM — train set (7200 samples)
SPLIT=train TOTAL=7200 bash slurm/submit_omni.sh

# Local (sequential, no SLURM)
bash run_local.sh

# Local — first 5 samples only
LIMIT=5 bash run_local.sh
```

## Output

```
results/
  validation/
    qwen25-omni-7b/
      summaries_task_0.jsonl
      ...
      summaries.jsonl
```

## Monitoring

```bash
squeue -u $USER
tail -f slurm/logs-slurm/slurm-Exp0002-omni-qwen25-7b_<JOB_ID>_<TASK>.out
```
