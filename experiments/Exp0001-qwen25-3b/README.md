# Exp0001 — Qwen2.5-Omni-3B

Lightweight-track baseline using the smallest Qwen Omni model.

| Field | Value |
|-------|-------|
| Model ID | `Qwen/Qwen2.5-Omni-3B` |
| Parameters | 3B (dense) |
| Family | Qwen2.5-Omni |
| Track | Lightweight |
| GPUs | 1 |
| Memory | 32 GB |
| Walltime/task | 30 min |

## Quick start

```bash
# From this directory:
cd experiments/Exp0001-qwen25-3b/

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
    qwen25-omni-3b/
      summaries_task_0.jsonl   # per-task outputs (SLURM)
      summaries_task_1.jsonl
      ...
      summaries.jsonl          # combined (after combine job)
```

## Monitoring

```bash
squeue -u $USER
tail -f slurm/logs-slurm/slurm-Exp0001-omni-qwen25-3b_<JOB_ID>_<TASK>.out
```
