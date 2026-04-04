#!/bin/bash

# Submit BeTraC Omni inference as a SLURM array job + combine job.
# Run from the repository root directory.
#
# Usage:
#   bash scripts/submit_slurm.sh                                  # HF dataset, defaults
#   bash scripts/submit_slurm.sh --manifest data/manifests/my.csv # custom data
#   bash scripts/submit_slurm.sh --time=04:00:00                  # override walltime
#
# Environment variables (all optional):
#   MODEL_ID          HuggingFace model name or local path  (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   MODEL_SHORT       Short name for output directory       (default: derived from MODEL_ID)
#   FLASH_ATTN        Enable Flash Attention 2              (default: 0)
#   THINKING          Enable thinking mode                  (default: 0)
#   SAMPLES_PER_TASK  Audio files per array task            (default: 4)
#
# Cluster-specific SBATCH options (--account, --partition, --time, etc.) can be
# passed as arguments and are forwarded to sbatch.
#
# Examples:
#   # Run on BeTraC HF validation set with default model
#   bash scripts/submit_slurm.sh
#
#   # Run with 3B model, custom account
#   MODEL_ID=Qwen/Qwen2.5-Omni-3B MODEL_SHORT=qwen25-3b \
#     bash scripts/submit_slurm.sh --account=MY_ACCOUNT
#
#   # Run thinking model on custom manifest
#   MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Thinking MODEL_SHORT=qwen3-thinking THINKING=1 \
#     bash scripts/submit_slurm.sh --manifest data/manifests/my_data.csv

set -e

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
FLASH_ATTN="${FLASH_ATTN:-0}"
THINKING="${THINKING:-0}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"

# Derive MODEL_SHORT from MODEL_ID if not set
if [ -z "${MODEL_SHORT}" ]; then
    MODEL_SHORT=$(basename "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')
fi

# ---------------------------------------------------------------------------
# Parse arguments: extract --manifest, pass rest to sbatch
# ---------------------------------------------------------------------------
MANIFEST=""
SBATCH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --manifest=*)
            MANIFEST="${arg#--manifest=}"
            ;;
        --manifest)
            # Next arg will be the path — handle in next iteration
            NEXT_IS_MANIFEST=1
            ;;
        *)
            if [ "${NEXT_IS_MANIFEST:-0}" = "1" ]; then
                MANIFEST="$arg"
                NEXT_IS_MANIFEST=0
            else
                SBATCH_ARGS+=("$arg")
            fi
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate run location
# ---------------------------------------------------------------------------
if [ ! -f "scripts/run_omni.slurm" ]; then
    echo "ERROR: Must run from the repository root directory."
    echo "  cd /path/to/betrac-2026-baseline"
    echo "  bash scripts/submit_slurm.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Determine data source and sample count
# ---------------------------------------------------------------------------
if [ -n "${MANIFEST}" ]; then
    if [ ! -f "${MANIFEST}" ]; then
        echo "ERROR: Manifest not found: ${MANIFEST}"
        exit 1
    fi
    SAMPLE_COUNT=$(tail -n +2 "${MANIFEST}" | wc -l | tr -d ' ')
    DATA_SOURCE="manifest: ${MANIFEST}"
else
    # Use HuggingFace dataset — need to count samples
    echo "Counting samples in BeTraC/betrac-2026 validation split..."
    SAMPLE_COUNT=$(steps/omni/.venv/bin/python -c "
from datasets import load_dataset
ds = load_dataset('BeTraC/betrac-2026', split='validation', streaming=True)
count = sum(1 for _ in ds)
print(count)
" 2>/dev/null)
    if [ -z "${SAMPLE_COUNT}" ] || [ "${SAMPLE_COUNT}" = "0" ]; then
        echo "ERROR: Could not count HuggingFace dataset samples."
        echo "Provide a manifest instead: bash scripts/submit_slurm.sh --manifest path/to/data.csv"
        exit 1
    fi
    DATA_SOURCE="HuggingFace: BeTraC/betrac-2026 [validation]"
fi

NUM_TASKS=$(( (SAMPLE_COUNT + SAMPLES_PER_TASK - 1) / SAMPLES_PER_TASK ))
MAX_TASK=$(( NUM_TASKS - 1 ))

# ---------------------------------------------------------------------------
# Create log directory
# ---------------------------------------------------------------------------
mkdir -p logs

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------
echo "=========================================="
echo "BeTraC Omni — SLURM Submission"
echo "=========================================="
echo "Data:             ${DATA_SOURCE}"
echo "Samples:          ${SAMPLE_COUNT}"
echo "Samples per task: ${SAMPLES_PER_TASK}"
echo "Array tasks:      0-${MAX_TASK}  (${NUM_TASKS} tasks)"
echo "Model ID:         ${MODEL_ID}"
echo "Model short:      ${MODEL_SHORT}"
echo "Flash Attn2:      ${FLASH_ATTN}"
echo "Thinking:         ${THINKING}"
echo "Extra sbatch args: ${SBATCH_ARGS[*]}"
echo ""

# ---------------------------------------------------------------------------
# Build export string
# ---------------------------------------------------------------------------
ARRAY_EXPORT="ALL,MODEL_ID=${MODEL_ID},MODEL_SHORT=${MODEL_SHORT}"
ARRAY_EXPORT+=",FLASH_ATTN=${FLASH_ATTN},THINKING=${THINKING}"
ARRAY_EXPORT+=",SAMPLES_PER_TASK=${SAMPLES_PER_TASK}"
if [ -n "${MANIFEST}" ]; then
    ARRAY_EXPORT+=",MANIFEST=$(realpath "${MANIFEST}")"
fi

# ---------------------------------------------------------------------------
# Submit array job
# ---------------------------------------------------------------------------
ARRAY_OUTPUT=$(sbatch \
    --array=0-${MAX_TASK} \
    --export="${ARRAY_EXPORT}" \
    "${SBATCH_ARGS[@]}" \
    scripts/run_omni.slurm)

echo "${ARRAY_OUTPUT}"
ARRAY_JOB_ID=$(echo "${ARRAY_OUTPUT}" | awk '{print $4}')

# ---------------------------------------------------------------------------
# Submit combine job (runs only after all array tasks succeed)
# ---------------------------------------------------------------------------
COMBINE_EXPORT="ALL,MODEL_SHORT=${MODEL_SHORT},MAX_TASK=${MAX_TASK}"

COMBINE_OUTPUT=$(sbatch \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export="${COMBINE_EXPORT}" \
    "${SBATCH_ARGS[@]}" \
    scripts/run_combine.slurm)

echo "${COMBINE_OUTPUT}"
COMBINE_JOB_ID=$(echo "${COMBINE_OUTPUT}" | awk '{print $4}')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Array job:       ${ARRAY_JOB_ID}  (tasks 0-${MAX_TASK})"
echo "Combine job:     ${COMBINE_JOB_ID}  (runs after all tasks succeed)"
echo ""
echo "Per-task output: results/${MODEL_SHORT}/summaries_task_N.jsonl"
echo "Combined output: results/${MODEL_SHORT}/summaries.jsonl"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm-betrac-omni_${ARRAY_JOB_ID}_<TASK>.out"
