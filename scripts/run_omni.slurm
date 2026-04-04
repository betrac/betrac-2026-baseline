#!/bin/bash
#SBATCH --job-name=betrac-omni
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x_%A_%a.out
#SBATCH --error=logs/slurm-%x_%A_%a.err

# BeTraC Omni inference — SLURM array job template
# Each task processes SAMPLES_PER_TASK audio files from the manifest.
# Results are combined by run_combine.slurm after all tasks complete.
#
# DO NOT submit directly — use submit_slurm.sh which calculates --array
# range and submits the combine job automatically.
#
# Environment variables (set by submit_slurm.sh via --export=):
#   MODEL_ID          HuggingFace model name or local path
#   MODEL_SHORT       Short name for output directory
#   MANIFEST          Path to CSV manifest file
#   SAMPLES_PER_TASK  Audio files per array task (default: 4)
#   FLASH_ATTN        Enable Flash Attention 2 (default: 0)
#   THINKING          Enable thinking mode (default: 0)
#
# Adapt for your cluster:
#   - Adjust --gpus-per-task, --mem, --time for your GPU type
#   - Add --account=YOUR_ACCOUNT and/or --partition=YOUR_PARTITION
#   - Add module load commands below if your cluster uses them

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
MODEL_SHORT="${MODEL_SHORT:-qwen3-omni-30b-instruct}"
FLASH_ATTN="${FLASH_ATTN:-0}"
THINKING="${THINKING:-0}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
BASELINE_DIR="${SLURM_SUBMIT_DIR}"
OMNI_DIR="${BASELINE_DIR}/steps/omni"
OMNI_VENV="${OMNI_DIR}/.venv"
OUTPUT_DIR="${BASELINE_DIR}/results/${MODEL_SHORT}"
TASK_JSONL="${OUTPUT_DIR}/summaries_task_${TASK_ID}.jsonl"

# ---------------------------------------------------------------------------
# Cluster-specific module loading (uncomment and adapt for your cluster)
# ---------------------------------------------------------------------------
# module purge
# module load python/3.11
# module load cuda/12.4

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job:              ${SLURM_JOB_NAME}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Task ID:          ${TASK_ID}"
echo "Node:             $(hostname)"
echo "Started at:       $(date)"
echo "Baseline:         ${BASELINE_DIR}"
echo "Model ID:         ${MODEL_ID}"
echo "Manifest:         ${MANIFEST}"
echo "Flash Attn2:      ${FLASH_ATTN}"
echo "Thinking:         ${THINKING}"
echo "Samples per task: ${SAMPLES_PER_TASK}"
echo "Task output:      ${TASK_JSONL}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}"
    exit 1
fi

if [ ! -f "${OMNI_VENV}/bin/python" ]; then
    echo "ERROR: Omni venv not found: ${OMNI_VENV}"
    echo "Run: cd ${BASELINE_DIR} && make setup"
    exit 1
fi

# ---------------------------------------------------------------------------
# Extract this task's manifest slice
# ---------------------------------------------------------------------------
FIRST_DATA_LINE=$(( TASK_ID * SAMPLES_PER_TASK + 2 ))
LAST_DATA_LINE=$(( FIRST_DATA_LINE + SAMPLES_PER_TASK - 1 ))

TEMP_MANIFEST=$(mktemp /tmp/manifest_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.XXXXXX.csv)
head -n 1 "${MANIFEST}" > "${TEMP_MANIFEST}"
sed -n "${FIRST_DATA_LINE},${LAST_DATA_LINE}p" "${MANIFEST}" >> "${TEMP_MANIFEST}"

TASK_SAMPLE_COUNT=$(tail -n +2 "${TEMP_MANIFEST}" | wc -l)

if [ "${TASK_SAMPLE_COUNT}" -eq 0 ]; then
    echo "No samples for task ${TASK_ID} (lines ${FIRST_DATA_LINE}-${LAST_DATA_LINE}). Exiting."
    rm -f "${TEMP_MANIFEST}"
    exit 0
fi

echo ""
echo "Task ${TASK_ID}: ${TASK_SAMPLE_COUNT} samples (manifest lines ${FIRST_DATA_LINE}-${LAST_DATA_LINE})"
tail -n +2 "${TEMP_MANIFEST}" | awk -F',' '{print "  " $1}'
echo ""

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "logs"

# ---------------------------------------------------------------------------
# Verify GPU
# ---------------------------------------------------------------------------
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""

# ---------------------------------------------------------------------------
# GPU utilization monitoring (background)
# ---------------------------------------------------------------------------
GPU_LOG="logs/gpu_util_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"
nvidia-smi dmon -s um -d 10 -o DT > "${GPU_LOG}" 2>/dev/null &
NVMON_PID=$!

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "Starting inference on ${TASK_SAMPLE_COUNT} samples..."

OMNI_ARGS=(
    --manifest "${TEMP_MANIFEST}"
    --output   "${TASK_JSONL}"
    --model    "${MODEL_ID}"
    --device   auto
    --base-dir "${BASELINE_DIR}"
    --resume
)

if [ "${FLASH_ATTN}" = "1" ]; then
    OMNI_ARGS+=(--use-flash-attn2)
    echo "Flash Attention 2: enabled"
fi

if [ "${THINKING}" = "1" ]; then
    OMNI_ARGS+=(--thinking)
    echo "Thinking mode: enabled"
fi

cd "${OMNI_DIR}"
"${OMNI_VENV}/bin/python" run_omni.py "${OMNI_ARGS[@]}"
EXIT_STATUS=$?

# ---------------------------------------------------------------------------
# Cleanup and report
# ---------------------------------------------------------------------------
kill "${NVMON_PID}" 2>/dev/null
rm -f "${TEMP_MANIFEST}"

echo ""
echo "=========================================="
if [ ${EXIT_STATUS} -eq 0 ]; then
    RESULT_COUNT=$(wc -l < "${TASK_JSONL}" 2>/dev/null || echo 0)
    echo "Task ${TASK_ID} SUCCESS: ${RESULT_COUNT}/${TASK_SAMPLE_COUNT} records"
    echo "Output: ${TASK_JSONL}"
else
    echo "Task ${TASK_ID} FAILED (exit code: ${EXIT_STATUS})"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit ${EXIT_STATUS}
