#!/bin/bash
# Local (non-SLURM) inference — processes all samples sequentially.
# Run from the repository root directory.
#
# Usage:
#   bash scripts/run_local.sh                                       # HF dataset, default model
#   bash scripts/run_local.sh --manifest data/manifests/my_data.csv # custom data
#   MODEL_ID=Qwen/Qwen2.5-Omni-3B bash scripts/run_local.sh       # different model

set -euo pipefail

# ---------------------------------------------------------------------------
# Default parameters (override via environment)
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
FLASH_ATTN="${FLASH_ATTN:-0}"
THINKING="${THINKING:-0}"

# Derive MODEL_SHORT from MODEL_ID if not set
if [ -z "${MODEL_SHORT:-}" ]; then
    MODEL_SHORT=$(basename "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')
fi

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OMNI_DIR="${BASELINE_DIR}/steps/omni"
OMNI_VENV="${OMNI_DIR}/.venv"
OUTPUT_DIR="${BASELINE_DIR}/results/${MODEL_SHORT}"
OUTPUT_JSONL="${OUTPUT_DIR}/summaries.jsonl"
LOG_DIR="${BASELINE_DIR}/logs"

# ---------------------------------------------------------------------------
# Parse arguments: extract --manifest
# ---------------------------------------------------------------------------
MANIFEST=""
for arg in "$@"; do
    case "$arg" in
        --manifest=*) MANIFEST="${arg#--manifest=}" ;;
        --manifest)   NEXT_IS_MANIFEST=1 ;;
        *)
            if [ "${NEXT_IS_MANIFEST:-0}" = "1" ]; then
                MANIFEST="$arg"
                NEXT_IS_MANIFEST=0
            fi
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================="
echo "BeTraC Omni — Local Inference"
echo "Host:             $(hostname)"
echo "Started at:       $(date)"
echo "Baseline:         ${BASELINE_DIR}"
echo "Model ID:         ${MODEL_ID}"
echo "Flash Attn2:      ${FLASH_ATTN}"
echo "Thinking:         ${THINKING}"
echo "Output:           ${OUTPUT_JSONL}"
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
# Build inference arguments
# ---------------------------------------------------------------------------
OMNI_ARGS=(
    --output "${OUTPUT_JSONL}"
    --model  "${MODEL_ID}"
    --device auto
    --resume
)

if [ -n "${MANIFEST}" ]; then
    if [ ! -f "${MANIFEST}" ]; then
        echo "ERROR: Manifest not found: ${MANIFEST}"
        exit 1
    fi
    OMNI_ARGS+=(--manifest "${MANIFEST}" --base-dir "${BASELINE_DIR}")
    SAMPLE_COUNT=$(tail -n +2 "${MANIFEST}" | wc -l | tr -d ' ')
    echo "Data source: manifest (${SAMPLE_COUNT} samples)"
else
    OMNI_ARGS+=(--dataset BeTraC/betrac-2026 --split validation)
    echo "Data source: HuggingFace BeTraC/betrac-2026 [validation]"
fi

if [ "${FLASH_ATTN}" = "1" ]; then
    OMNI_ARGS+=(--use-flash-attn2)
fi

if [ "${THINKING}" = "1" ]; then
    OMNI_ARGS+=(--thinking)
fi

echo ""

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# GPU info and monitoring (optional)
# ---------------------------------------------------------------------------
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no nvidia-smi — using CPU or MPS)"
echo ""

GPU_LOG="${LOG_DIR}/gpu_util_$(date +%Y%m%d_%H%M%S).log"
nvidia-smi dmon -s um -d 10 -o DT > "${GPU_LOG}" 2>/dev/null &
NVMON_PID=$!

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "Starting inference..."

cd "${OMNI_DIR}"
"${OMNI_VENV}/bin/python" run_omni.py "${OMNI_ARGS[@]}"
EXIT_STATUS=$?

# ---------------------------------------------------------------------------
# Cleanup and report
# ---------------------------------------------------------------------------
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
