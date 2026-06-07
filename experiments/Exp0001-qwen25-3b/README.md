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

## Eval set

The eval dataset (`BeTraC/betrac-2026-eval`) is private. Teams must be granted
access during the challenge. Once approved, export your HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here   # or add to ~/.bashrc
```

The submit script pre-caches the dataset **and model weights** on the login
node before submitting jobs (requires internet access). Set `SKIP_CACHE=1` to
skip this if already cached from a previous run.

```bash
# SLURM (875 eval samples, 20 per task)
DATASET=BeTraC/betrac-2026-eval SPLIT=eval TOTAL=875 SAMPLES_PER_TASK=20 \
  bash slurm/submit_omni.sh

# Local (sequential, no SLURM)
DATASET=BeTraC/betrac-2026-eval SPLIT=eval bash run_local.sh
```

`TOTAL=875` skips the slow auto-count; omit it to auto-detect the sample count.
Results go to `results/eval/qwen25-omni-3b/summaries.jsonl`.
See [cmd.txt](cmd.txt) for more command variations.

## Output

```
results/
  validation/                          # public validation set
    qwen25-omni-3b/
      summaries_task_0.jsonl
      ...
      summaries.jsonl                  # combined
  eval/                                # eval set (after access grant)
    qwen25-omni-3b/
      summaries.jsonl
```

## Monitoring

```bash
squeue -u $USER
tail -f slurm/logs-slurm/slurm-Exp0001-omni-qwen25-3b_<JOB_ID>_<TASK>.out
```
