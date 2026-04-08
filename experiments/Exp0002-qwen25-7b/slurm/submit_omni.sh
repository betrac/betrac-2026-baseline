#!/bin/bash

# Submit Qwen2.5-Omni-7B inference array job + combine job for Exp0002
# Run from: experiments/Exp0002-qwen25-7b/
#
# Usage:
#   bash slurm/submit_omni.sh                        # defaults (validation, 400 samples)
#   TOTAL=400 bash slurm/submit_omni.sh              # skip slow HF count
#   SPLIT=train TOTAL=7200 bash slurm/submit_omni.sh # train split
#   FLASH_ATTN=1 bash slurm/submit_omni.sh           # enable Flash Attention 2
#   bash slurm/submit_omni.sh --time=02:00:00        # override per-task walltime
#
# Environment variables (all optional):
#   MODEL_ID          HuggingFace model ID           (default: Qwen/Qwen2.5-Omni-7B)
#   MODEL_SHORT       Short name for output dir      (default: qwen25-omni-7b)
#   DATASET           HF dataset name                (default: BeTraC/betrac-2026)
#   SPLIT             Dataset split                  (default: validation)
#   TOTAL             Total sample count (skip HF count) (default: auto-detect)
#   FLASH_ATTN        Enable Flash Attention 2       (default: 0)
#   SAMPLES_PER_TASK  Samples per array task         (default: 4)
#
# All positional arguments are forwarded to the array job sbatch call.

set -e

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Omni-7B}"
MODEL_SHORT="${MODEL_SHORT:-qwen25-omni-7b}"
DATASET="${DATASET:-BeTraC/betrac-2026}"
SPLIT="${SPLIT:-validation}"
FLASH_ATTN="${FLASH_ATTN:-0}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"

# ---------------------------------------------------------------------------
# Validate run location
# ---------------------------------------------------------------------------
if [ ! -f "slurm/run_omni_hf.slurm" ]; then
    echo "ERROR: Must run from experiments/Exp0002-qwen25-7b/"
    echo "  cd $(cd ../.. && pwd)/experiments/Exp0002-qwen25-7b"
    echo "  bash slurm/submit_omni.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
EXPERIMENT_DIR="$(pwd)"
BASELINE_DIR="$(cd "../.." && pwd)"
OMNI_VENV="${BASELINE_DIR}/steps/omni/.venv"

# ---------------------------------------------------------------------------
# Pre-cache dataset and model (avoids 429 rate-limit errors in parallel tasks)
# ---------------------------------------------------------------------------
SKIP_CACHE="${SKIP_CACHE:-0}"
if [ "${SKIP_CACHE}" != "1" ]; then
    echo "Pre-caching dataset and model (set SKIP_CACHE=1 to skip)..."
    "${OMNI_VENV}/bin/python" -c "
from datasets import load_dataset
from huggingface_hub import snapshot_download
print('Caching dataset: ${DATASET} [${SPLIT}]...')
ds = load_dataset('${DATASET}', split='${SPLIT}')
print(f'  Dataset cached: {len(ds)} samples')
print('Caching model: ${MODEL_ID}...')
snapshot_download('${MODEL_ID}')
print('  Model cached.')
"
    if [ $? -ne 0 ]; then
        echo "WARNING: Pre-caching failed. Jobs may hit HF API rate limits."
    fi
fi

# ---------------------------------------------------------------------------
# Count samples
# ---------------------------------------------------------------------------
SAMPLE_COUNT="${TOTAL:-}"
if [ -z "${SAMPLE_COUNT}" ]; then
    echo "Counting samples in ${DATASET} [${SPLIT}] (set TOTAL=N to skip)..."
    SAMPLE_COUNT=$("${OMNI_VENV}/bin/python" -c "
from datasets import load_dataset
ds = load_dataset('${DATASET}', split='${SPLIT}', streaming=True)
print(sum(1 for _ in ds))
" 2>/dev/null)
    if [ -z "${SAMPLE_COUNT}" ] || [ "${SAMPLE_COUNT}" = "0" ]; then
        echo "ERROR: Could not count HuggingFace dataset samples."
        echo "Set TOTAL=N manually, e.g.: TOTAL=400 bash slurm/submit_omni.sh"
        exit 1
    fi
fi

NUM_TASKS=$(( (SAMPLE_COUNT + SAMPLES_PER_TASK - 1) / SAMPLES_PER_TASK ))
MAX_TASK=$(( NUM_TASKS - 1 ))

# ---------------------------------------------------------------------------
# Create log directory
# ---------------------------------------------------------------------------
mkdir -p slurm/logs-slurm

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Exp0002 — Qwen2.5-Omni-7B Submission"
echo "=========================================="
echo "Dataset:          ${DATASET} [${SPLIT}]"
echo "Samples:          ${SAMPLE_COUNT}"
echo "Samples per task: ${SAMPLES_PER_TASK}"
echo "Array tasks:      0-${MAX_TASK}  (${NUM_TASKS} tasks)"
echo "Model ID:         ${MODEL_ID}"
echo "Model short:      ${MODEL_SHORT}"
echo "Flash Attn2:      ${FLASH_ATTN}"
echo "Extra args:       $*"
echo ""

# ---------------------------------------------------------------------------
# Submit array job
# ---------------------------------------------------------------------------
ARRAY_EXPORT="ALL,MODEL_ID=${MODEL_ID},MODEL_SHORT=${MODEL_SHORT}"
ARRAY_EXPORT+=",DATASET=${DATASET},SPLIT=${SPLIT}"
ARRAY_EXPORT+=",FLASH_ATTN=${FLASH_ATTN},SAMPLES_PER_TASK=${SAMPLES_PER_TASK}"

ARRAY_OUTPUT=$(sbatch \
    --array=0-${MAX_TASK} \
    --export="${ARRAY_EXPORT}" \
    "$@" \
    slurm/run_omni_hf.slurm)

echo "${ARRAY_OUTPUT}"
ARRAY_JOB_ID=$(echo "${ARRAY_OUTPUT}" | awk '{print $4}')

# ---------------------------------------------------------------------------
# Submit combine job
# ---------------------------------------------------------------------------
COMBINE_EXPORT="ALL,MODEL_SHORT=${MODEL_SHORT},SPLIT=${SPLIT},MAX_TASK=${MAX_TASK}"

COMBINE_OUTPUT=$(sbatch \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export="${COMBINE_EXPORT}" \
    slurm/run_combine_omni.slurm)

echo "${COMBINE_OUTPUT}"
COMBINE_JOB_ID=$(echo "${COMBINE_OUTPUT}" | awk '{print $4}')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Array job:   ${ARRAY_JOB_ID}  (tasks 0-${MAX_TASK})"
echo "Combine job: ${COMBINE_JOB_ID}  (runs after all tasks succeed)"
echo ""
echo "Per-task outputs:  results/${SPLIT}/${MODEL_SHORT}/summaries_task_N.jsonl"
echo "Combined output:   results/${SPLIT}/${MODEL_SHORT}/summaries.jsonl"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs-slurm/slurm-Exp0002-omni-qwen25-7b_${ARRAY_JOB_ID}_<TASK>.out"
