#!/bin/bash
#SBATCH --job-name=Exp0002-omni-combine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --cluster=ascend                    # ← CHANGE: your cluster or --partition=gpu
#SBATCH --account=PAS2138                   # ← CHANGE: your account/allocation
#SBATCH --output=slurm/logs-slurm/slurm-%x_%j.out
#SBATCH --error=slurm/logs-slurm/slurm-%x_%j.err

# Combine per-task JSONL files into a single summaries.jsonl (Exp0002)
# Submitted automatically by submit_omni.sh with --dependency=afterok:<ARRAY_JOB_ID>.
#
# Environment variables (set by submit_omni.sh via --export=):
#   MODEL_SHORT  Short name matching the output directory (default: qwen25-omni-7b)
#   SPLIT        Dataset split (default: validation)
#   MAX_TASK     Highest array task index

MODEL_SHORT="${MODEL_SHORT:-qwen25-omni-7b}"
SPLIT="${SPLIT:-validation}"
MAX_TASK="${MAX_TASK:-0}"

EXPERIMENT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${EXPERIMENT_DIR}/results/${SPLIT}/${MODEL_SHORT}"
COMBINED_JSONL="${OUTPUT_DIR}/summaries.jsonl"

echo "=========================================="
echo "Job:        ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started at: $(date)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Tasks:      0-${MAX_TASK}"
echo "=========================================="

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

echo ""
echo "Combining $((MAX_TASK + 1)) task files into ${COMBINED_JSONL}..."

TEMP_COMBINED=$(mktemp "${OUTPUT_DIR}/summaries_combined.XXXXXX.jsonl")

TOTAL=0
for TASK_ID in $(seq 0 "${MAX_TASK}"); do
    TASK_JSONL="${OUTPUT_DIR}/summaries_task_${TASK_ID}.jsonl"
    COUNT=$(wc -l < "${TASK_JSONL}")
    cat "${TASK_JSONL}" >> "${TEMP_COMBINED}"
    TOTAL=$(( TOTAL + COUNT ))
    echo "  task ${TASK_ID}: appended ${COUNT} records"
done

# Deduplicate by sample ID — keeps the last occurrence (most recent result).
# This handles retries with different SAMPLES_PER_TASK where task files overlap.
python3 -c "
import json, sys
seen = {}
for line in open('${TEMP_COMBINED}'):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    seen[r['id']] = line
with open('${COMBINED_JSONL}', 'w') as f:
    for line in seen.values():
        f.write(line + '\n')
print(f'Deduplicated: {len(seen)} unique records')
"
DEDUP_STATUS=$?
rm -f "${TEMP_COMBINED}"

if [ ${DEDUP_STATUS} -ne 0 ]; then
    echo "WARNING: Deduplication failed, falling back to raw concatenation"
    cat "${OUTPUT_DIR}"/summaries_task_*.jsonl > "${COMBINED_JSONL}"
fi

COMBINED_COUNT=$(wc -l < "${COMBINED_JSONL}")
echo ""
echo "Combined: ${COMBINED_COUNT} unique records (from ${TOTAL} total lines)"

if [ "${COMBINED_COUNT}" -ne "${TOTAL}" ]; then
    echo "NOTE: ${TOTAL} - ${COMBINED_COUNT} = $(( TOTAL - COMBINED_COUNT )) duplicates removed"
fi

echo ""
echo "=========================================="
echo "SUCCESS: ${COMBINED_COUNT} records → ${COMBINED_JSONL}"
echo "Completed at: $(date)"
echo "=========================================="
