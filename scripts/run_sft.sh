#!/usr/bin/env bash
set -euo pipefail

# Train one model snapshot. Model serving and evaluation are configured separately.
# Usage: ./run_sft.sh <model_type> <model> <dataset> <output_dir>

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <model_type> <model> <dataset> <output_dir>"
    echo "  model_type: 7b or 72b"
    exit 1
fi

MODEL_TYPE="$1"
MODEL="$(realpath "$2")"
DATASET="$(realpath "$3")"
OUTPUT_DIR="$4"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${MODEL_TYPE}" in
    7b) TRAIN_SCRIPT="${SCRIPT_DIR}/run_swift_sft_7b.sh" ;;
    72b) TRAIN_SCRIPT="${SCRIPT_DIR}/run_swift_sft_72b.sh" ;;
    *)
        echo "Error: model_type must be 7b or 72b"
        exit 1
        ;;
esac

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Error: training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

export MODEL DATASET OUTPUT_DIR
bash "${TRAIN_SCRIPT}" > "${OUTPUT_DIR}/train.log" 2>&1
echo "Training completed. Outputs: ${OUTPUT_DIR}"
