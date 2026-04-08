# Exp0004 — Qwen3-Omni-30B-A3B-Thinking

Heavyweight-track baseline using the thinking variant of Qwen3-Omni MoE. Generates chain-of-thought reasoning in `<think>` blocks before producing the SOAP note.

| Field | Value |
|-------|-------|
| Model ID | `Qwen/Qwen3-Omni-30B-A3B-Thinking` |
| Parameters | 30B total / ~3B active (MoE) |
| Family | Qwen3-Omni-MoE |
| Track | Heavyweight |
| GPUs (SBATCH) | 2 (1 used + CPU offload) |
| Memory | 80 GB |
| Time/sample | ~21 min (median on A100-40GB) |
| Walltime/task | 2 hours (4 samples/task) |
| Extra flags | `--thinking` |

## Quick start

```bash
cd experiments/Exp0004-qwen3-thinking/

# SLURM — validation set (400 samples)
TOTAL=400 bash slurm/submit_omni.sh

# SLURM — train set (7200 samples)
SPLIT=train TOTAL=7200 bash slurm/submit_omni.sh

# Local (sequential, no SLURM)
bash run_local.sh

# Local — first 5 samples only
LIMIT=5 bash run_local.sh
```

## Notes

- Thinking mode is enabled by default (`THINKING=1`). The model auto-detects thinking mode from the model name, but the `--thinking` flag is set explicitly for clarity.
- Output JSONL includes a `thinking` field with the chain-of-thought content.
- Inference is ~3x slower than the Instruct variant due to chain-of-thought generation (median ~21 min vs ~7 min per sample). Some samples can take 40+ min.
- `thinker_max_new_tokens` is set to 8192 (observed max thinking output: ~4,600 tokens).

## Output

```
results/
  validation/
    qwen3-omni-30b-thinking/
      summaries_task_0.jsonl
      ...
      summaries.jsonl
```

## Monitoring

```bash
squeue -u $USER
tail -f slurm/logs-slurm/slurm-Exp0004-omni-qwen3-thinking_<JOB_ID>_<TASK>.out
```
