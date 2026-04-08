# Exp0003 — Qwen3-Omni-30B-A3B-Instruct

Heavyweight-track baseline using the default MoE Qwen3-Omni model (30B total, ~3B active params).

| Field | Value |
|-------|-------|
| Model ID | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| Parameters | 30B total / ~3B active (MoE) |
| Family | Qwen3-Omni-MoE |
| Track | Heavyweight |
| GPUs (SBATCH) | 2 (1 used + CPU offload) |
| Memory | 80 GB |
| Time/sample | ~7 min (median on A100-40GB) |
| Walltime/task | 1.5 hours (4 samples/task) |

## Quick start

```bash
cd experiments/Exp0003-qwen3-instruct/

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
    qwen3-omni-30b-instruct/
      summaries_task_0.jsonl
      ...
      summaries.jsonl
```

## Monitoring

```bash
squeue -u $USER
tail -f slurm/logs-slurm/slurm-Exp0003-omni-qwen3-instruct_<JOB_ID>_<TASK>.out
```
