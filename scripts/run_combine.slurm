#!/bin/bash
#SBATCH --job-name=betrac-combine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%x_%j.out
#SBATCH --error=logs/slurm-%x_%j.err

# Combine per-task JSONL files into a single summaries.jsonl
# Submitted automatically by submit_slurm.sh with --dependency=afterok:<ARRAY_JOB_ID>.
#
# Environment variables (set by submit_slurm.sh via --export=):
#   MODEL_SHORT  Short name matching the output directory
#   MAX_TASK     Highest array task index (e.g. 4 for a 5-task array)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
MODEL_SHORT="${MODEL_SHORT:-qwen3-omni-30b-instruct}"
MAX_TASK="${MAX_TASK:-0}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${BASELINE_DIR}/results/${MODEL_SHORT}"
COMBINED_JSONL="${OUTPUT_DIR}/summaries.jsonl"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job:        ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started at: $(date)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Tasks:      0-${MAX_TASK}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Validate all task outputs exist
# ---------------------------------------------------------------------------
echo ""
echo "Checking task outputs..."
MISSING=()
for TASK_ID in $(seq 0 "${MAX_TASK}"); do
    TASK_JSONL="${OUTPUT_DIR}/summaries_task_${TASK_ID}.jsonl"
    if [ ! -f "${TASK_JSONL}" ]; then
        MISSING+=("${TASK_JSONL}")
        echo "  MISSING: ${TASK_JSONL}"
    else
        COUNT=$(wc -l < "${TASK_JSONL}")
        echo "  OK task ${TASK_ID}: ${COUNT} records"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: ${#MISSING[@]} task output(s) missing. Cannot combine."
    exit 1
fi

# ---------------------------------------------------------------------------
# Combine in task order
# ---------------------------------------------------------------------------
echo ""
echo "Combining $((MAX_TASK + 1)) task files into ${COMBINED_JSONL}..."

rm -f "${COMBINED_JSONL}"

TOTAL=0
for TASK_ID in $(seq 0 "${MAX_TASK}"); do
    TASK_JSONL="${OUTPUT_DIR}/summaries_task_${TASK_ID}.jsonl"
    COUNT=$(wc -l < "${TASK_JSONL}")
    cat "${TASK_JSONL}" >> "${COMBINED_JSONL}"
    TOTAL=$(( TOTAL + COUNT ))
    echo "  task ${TASK_ID}: appended ${COUNT} records"
done

COMBINED_COUNT=$(wc -l < "${COMBINED_JSONL}")
echo ""
echo "Combined: ${COMBINED_COUNT} total records"

if [ "${COMBINED_COUNT}" -ne "${TOTAL}" ]; then
    echo "WARNING: combined count (${COMBINED_COUNT}) != sum of task counts (${TOTAL})"
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "SUCCESS: ${COMBINED_COUNT} records → ${COMBINED_JSONL}"
echo "Completed at: $(date)"
echo "=========================================="
