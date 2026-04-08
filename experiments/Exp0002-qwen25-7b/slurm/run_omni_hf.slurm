#!/bin/bash
#SBATCH --job-name=Exp0002-omni-qwen25-7b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --cluster=ascend                    # ← CHANGE: your cluster or --partition=gpu
#SBATCH --account=PAS2138                   # ← CHANGE: your account/allocation
#SBATCH --output=slurm/logs-slurm/slurm-%x_%A_%a.out
#SBATCH --error=slurm/logs-slurm/slurm-%x_%A_%a.err

# Exp0002: Qwen2.5-Omni-7B inference on BeTraC HF dataset — array job
# Each task processes SAMPLES_PER_TASK samples via --offset/--limit.
# Results combined by run_combine_omni.slurm after all tasks complete.
#
# DO NOT submit directly — use submit_omni.sh which sets --array and
# submits the combine job automatically.
#
# Environment variables (set by submit_omni.sh via --export=):
#   MODEL_ID          HuggingFace model ID       (default: Qwen/Qwen2.5-Omni-7B)
#   MODEL_SHORT       Short name for output dir   (default: qwen25-omni-7b)
#   DATASET           HF dataset name             (default: BeTraC/betrac-2026)
#   SPLIT             Dataset split               (default: validation)
#   FLASH_ATTN        Enable Flash Attention 2    (default: 0)
#   SAMPLES_PER_TASK  Samples per array task      (default: 4)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Omni-7B}"
MODEL_SHORT="${MODEL_SHORT:-qwen25-omni-7b}"
DATASET="${DATASET:-BeTraC/betrac-2026}"
SPLIT="${SPLIT:-validation}"
FLASH_ATTN="${FLASH_ATTN:-0}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
EXPERIMENT_DIR="${SLURM_SUBMIT_DIR}"
BASELINE_DIR="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
OMNI_DIR="${BASELINE_DIR}/steps/omni"
OMNI_VENV="${OMNI_DIR}/.venv"
OUTPUT_DIR="${EXPERIMENT_DIR}/results/${SPLIT}/${MODEL_SHORT}"
TASK_JSONL="${OUTPUT_DIR}/summaries_task_${TASK_ID}.jsonl"
LOG_DIR="${EXPERIMENT_DIR}/slurm/logs-slurm"

# ---------------------------------------------------------------------------
# Calculate offset for this task
# ---------------------------------------------------------------------------
OFFSET=$(( TASK_ID * SAMPLES_PER_TASK ))

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
echo "Dataset:          ${DATASET} [${SPLIT}]"
echo "Offset:           ${OFFSET}"
echo "Limit:            ${SAMPLES_PER_TASK}"
echo "Flash Attn2:      ${FLASH_ATTN}"
echo "Task output:      ${TASK_JSONL}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [ ! -f "${OMNI_VENV}/bin/python" ]; then
    echo "ERROR: Omni venv not found: ${OMNI_VENV}"
    echo "Run: cd ${BASELINE_DIR} && make setup"
    exit 1
fi

# ---------------------------------------------------------------------------
# Load modules and environment
# ---------------------------------------------------------------------------
module purge                                # ← CHANGE: adapt to your cluster's
module load python/3.10                     #   module system, conda, or container
module load cuda/12.4.1                     #   setup (see SLURM_SETUP.md §3)

export PATH="$HOME/.local/bin:$PATH"

# Use cached HF data only — avoids 429 rate-limit errors from parallel tasks
export HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Verify GPU
# ---------------------------------------------------------------------------
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""

# ---------------------------------------------------------------------------
# GPU utilization monitoring (background)
# ---------------------------------------------------------------------------
GPU_LOG="${LOG_DIR}/gpu_util_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"
nvidia-smi dmon -s um -d 10 -o DT > "${GPU_LOG}" 2>/dev/null &
NVMON_PID=$!

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "Starting inference: offset=${OFFSET}, limit=${SAMPLES_PER_TASK}..."

OMNI_ARGS=(
    --dataset  "${DATASET}"
    --split    "${SPLIT}"
    --offset   "${OFFSET}"
    --limit    "${SAMPLES_PER_TASK}"
    --output   "${TASK_JSONL}"
    --model    "${MODEL_ID}"
    --device   auto
    --resume
)

if [ "${FLASH_ATTN}" = "1" ]; then
    OMNI_ARGS+=(--use-flash-attn2)
    echo "Flash Attention 2: enabled"
fi

cd "${OMNI_DIR}"
"${OMNI_VENV}/bin/python" run_omni.py "${OMNI_ARGS[@]}"
EXIT_STATUS=$?

# ---------------------------------------------------------------------------
# Cleanup and report
# ---------------------------------------------------------------------------
kill "${NVMON_PID}" 2>/dev/null
echo "GPU utilization log: ${GPU_LOG}"

echo ""
echo "=========================================="
if [ ${EXIT_STATUS} -eq 0 ]; then
    RESULT_COUNT=$(wc -l < "${TASK_JSONL}" 2>/dev/null || echo 0)
    echo "Task ${TASK_ID} SUCCESS: ${RESULT_COUNT} records"
    echo "Output: ${TASK_JSONL}"
else
    echo "Task ${TASK_ID} FAILED (exit code: ${EXIT_STATUS})"
    echo "Check: ${LOG_DIR}/slurm-${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit ${EXIT_STATUS}
