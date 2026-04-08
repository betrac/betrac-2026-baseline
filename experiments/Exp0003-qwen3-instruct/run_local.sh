#!/bin/bash
# Local (non-SLURM) inference for Exp0003 — Qwen3-Omni-30B-A3B-Instruct
# Processes all samples sequentially using HF dataset mode.
#
# Usage:
#   ./experiments/Exp0003-qwen3-instruct/run_local.sh
#   SPLIT=train ./experiments/Exp0003-qwen3-instruct/run_local.sh
#   LIMIT=5 ./experiments/Exp0003-qwen3-instruct/run_local.sh

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
MODEL_SHORT="${MODEL_SHORT:-qwen3-omni-30b-instruct}"
DATASET="${DATASET:-BeTraC/betrac-2026}"
SPLIT="${SPLIT:-validation}"
LIMIT="${LIMIT:-}"
FLASH_ATTN="${FLASH_ATTN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}"
BASELINE_DIR="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
OMNI_DIR="${BASELINE_DIR}/steps/omni"
OMNI_VENV="${OMNI_DIR}/.venv"
OUTPUT_DIR="${EXPERIMENT_DIR}/results/${SPLIT}/${MODEL_SHORT}"
OUTPUT_JSONL="${OUTPUT_DIR}/summaries.jsonl"
LOG_DIR="${EXPERIMENT_DIR}/logs-local"

echo "=========================================="
echo "Local run:        Exp0003 — Qwen3-Omni-30B Instruct"
echo "Host:             $(hostname)"
echo "Started at:       $(date)"
echo "Baseline:         ${BASELINE_DIR}"
echo "Model ID:         ${MODEL_ID}"
echo "Dataset:          ${DATASET} [${SPLIT}]"
echo "Flash Attn2:      ${FLASH_ATTN}"
echo "Output:           ${OUTPUT_JSONL}"
echo "=========================================="

if [ ! -f "${OMNI_VENV}/bin/python" ]; then
    echo "ERROR: Omni venv not found: ${OMNI_VENV}"
    echo "Run: cd ${BASELINE_DIR} && make setup"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no nvidia-smi — using CPU or MPS)"
echo ""

GPU_LOG="${LOG_DIR}/gpu_util_$(date +%Y%m%d_%H%M%S).log"
nvidia-smi dmon -s um -d 10 -o DT > "${GPU_LOG}" 2>/dev/null &
NVMON_PID=$!

OMNI_ARGS=(
    --dataset "${DATASET}"
    --split   "${SPLIT}"
    --output  "${OUTPUT_JSONL}"
    --model   "${MODEL_ID}"
    --device  auto
    --resume
)

if [ -n "${LIMIT}" ]; then
    OMNI_ARGS+=(--limit "${LIMIT}")
fi

if [ "${FLASH_ATTN}" = "1" ]; then
    OMNI_ARGS+=(--use-flash-attn2)
    echo "Flash Attention 2: enabled"
fi

echo "Starting inference..."

cd "${OMNI_DIR}"
"${OMNI_VENV}/bin/python" run_omni.py "${OMNI_ARGS[@]}"
EXIT_STATUS=$?

kill "${NVMON_PID}" 2>/dev/null || true

echo ""
echo "=========================================="
if [ ${EXIT_STATUS} -eq 0 ]; then
    RESULT_COUNT=$(wc -l < "${OUTPUT_JSONL}" 2>/dev/null || echo 0)
    echo "SUCCESS: ${RESULT_COUNT} records → ${OUTPUT_JSONL}"
else
    echo "FAILED (exit code: ${EXIT_STATUS})"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit ${EXIT_STATUS}
