#!/bin/bash
set -euo pipefail

# OVS (Observation-Variation-Selection) Data Evolution
# Performs multiple rounds of data quality enhancement before training
#
# Required environment variables (set by run.sh):
#   OVS_INPUT_DIR     - Input directory containing .jsonl files
#   OVS_OUTPUT_DIR    - Output directory for evolved data
#   OVS_MODEL         - Model name for LLM calls
#   OVS_API_KEY       - API key for model access
#   OVS_SERVICE_URLS  - Semicolon-separated list of service URLs
#   OVS_LOG_FILE      - Path to log file for monitoring
#
# Optional environment variables:
#   OVS_MAX_RETRY     - Maximum retry attempts to complete missing samples (default: 3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVS_DIR="$(cd "${SCRIPT_DIR}/../src/ovs" && pwd)"
ORIGINAL_DIR="$(pwd)"

# =============== Configuration Validation ===============
# Validate required environment variables
: "${OVS_INPUT_DIR:?Error: OVS_INPUT_DIR not set}"
: "${OVS_OUTPUT_DIR:?Error: OVS_OUTPUT_DIR not set}"
: "${OVS_MODEL:?Error: OVS_MODEL not set}"
: "${OVS_API_KEY:?Error: OVS_API_KEY not set}"
: "${OVS_SERVICE_URLS:?Error: OVS_SERVICE_URLS not set}"
: "${OVS_LOG_FILE:?Error: OVS_LOG_FILE not set}"

# =============== Convert to Absolute Paths ===============
cd "${ORIGINAL_DIR}"
OVS_INPUT_DIR="$(cd "${OVS_INPUT_DIR}" && pwd)"
mkdir -p "${OVS_OUTPUT_DIR}"
OVS_OUTPUT_DIR="$(cd "${OVS_OUTPUT_DIR}" && pwd)"
mkdir -p "$(dirname "${OVS_LOG_FILE}")"
touch "${OVS_LOG_FILE}"
OVS_LOG_FILE="$(cd "$(dirname "${OVS_LOG_FILE}")" && pwd)/$(basename "${OVS_LOG_FILE}")"

# Export validated variables with absolute paths
export OVS_INPUT_DIR
export OVS_OUTPUT_DIR
export OVS_MODEL
export OVS_API_KEY
export OVS_SERVICE_URLS
export OVS_PROMPT_LANGUAGE="${OVS_PROMPT_LANGUAGE:-zh}"

# =============== Implementation Details ===============
# These are typically fixed values, override via env vars if needed
export OVS_ROUNDS="${OVS_ROUNDS:-4}"
export OVS_REWRITE_ROUNDS="${OVS_REWRITE_ROUNDS:-2}"
export OVS_CONCURRENT="${OVS_CONCURRENT:-500}"
export OVS_TIMEOUT="${OVS_TIMEOUT:-720}"
export OVS_MIN_DIFF_MAIN="${OVS_MIN_DIFF_MAIN:-0.05}"
export OVS_MIN_DIFF_LIGHT="${OVS_MIN_DIFF_LIGHT:-0.03}"
export OVS_ERROR_LOG_INTERVAL="${OVS_ERROR_LOG_INTERVAL:-10}"
export OVS_MAX_RETRY="${OVS_MAX_RETRY:-3}"

# =============== Utility Functions ===============
# Count lines in a file
count_lines() {
    local file="$1"
    if [[ -f "$file" ]]; then
        wc -l < "$file" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# =============== Find Input Data Files ===============
INPUT_FILES=()
while IFS= read -r -d '' file; do
    INPUT_FILES+=("$file")
done < <(find "${OVS_INPUT_DIR}" -name "*.jsonl" -type f -print0 | sort -z)

# =============== Header ===============
echo ""
echo "========================================"
echo "OVS Data Evolution"
echo "========================================"
echo "Input directory: ${OVS_INPUT_DIR}"
echo "Output directory: ${OVS_OUTPUT_DIR}"
echo "Input files: ${#INPUT_FILES[@]}"
echo "Evolution rounds: ${OVS_ROUNDS}"
echo "Concurrent requests: ${OVS_CONCURRENT}"
echo "Max retry attempts: ${OVS_MAX_RETRY}"
echo "Model: ${OVS_MODEL}"
echo "Monitor: tail -f ${OVS_LOG_FILE}"
echo ""

# Check if input files exist
if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Error: No jsonl files found in ${OVS_INPUT_DIR}"
    exit 1
fi

# =============== Run OVS with Retry Logic ===============
cd "${OVS_DIR}"

RETRY_COUNT=0
ALL_COMPLETE=false

while [[ ${RETRY_COUNT} -lt ${OVS_MAX_RETRY} ]]; do
    if [[ ${RETRY_COUNT} -gt 0 ]]; then
        echo ""
        echo "========================================"
        echo "Retry Attempt ${RETRY_COUNT}/${OVS_MAX_RETRY}"
        echo "========================================"
        echo "Some samples may have failed in previous attempts"
        echo "Re-running to process missing samples..."
        echo ""
    fi

    # Run OVS
    python3 ovs.py >> "${OVS_LOG_FILE}" 2>&1
    OVS_EXIT_CODE=$?

    # Count input and output lines for each file
    ALL_COMPLETE=true
    for input_file in "${INPUT_FILES[@]}"; do
        base_name=$(basename "${input_file}" .jsonl)
        input_lines=$(count_lines "${input_file}")

        # Check final round output (new structure: round4/data/base_name.jsonl)
        output_file="${OVS_OUTPUT_DIR}/round${OVS_ROUNDS}/data/${base_name}.jsonl"
        output_lines=$(count_lines "${output_file}")

        echo "[$(basename ${input_file})] Input: ${input_lines} lines, Output: ${output_lines} lines"

        if [[ ${output_lines} -lt ${input_lines} ]]; then
            echo "  Warning: Missing $((input_lines - output_lines)) samples"
            ALL_COMPLETE=false
        fi
    done

    # If all files are complete, break
    if [[ "${ALL_COMPLETE}" == "true" ]]; then
        echo ""
        echo "All samples processed successfully!"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))

    # If we haven't reached max retry, continue
    if [[ ${RETRY_COUNT} -lt ${OVS_MAX_RETRY} ]]; then
        echo ""
        echo "Some samples are still missing. Will retry in 5 seconds..."
        sleep 5
    fi
done

# =============== Final Summary ===============
echo ""
echo "========================================"
echo "OVS Data Evolution Summary"
echo "========================================"

for input_file in "${INPUT_FILES[@]}"; do
    base_name=$(basename "${input_file}" .jsonl)
    input_lines=$(count_lines "${input_file}")
    output_file="${OVS_OUTPUT_DIR}/round${OVS_ROUNDS}/data/${base_name}.jsonl"
    output_lines=$(count_lines "${output_file}")
    missing=$((input_lines - output_lines))

    echo ""
    echo "File: ${base_name}"
    echo "  Input samples:  ${input_lines}"
    echo "  Output samples: ${output_lines}"
    echo "  Missing:        ${missing}"

    if [[ ${missing} -gt 0 ]]; then
        echo "  Status:         INCOMPLETE (${missing} samples missing after ${RETRY_COUNT} retries)"
    else
        echo "  Status:         COMPLETE"
    fi
done

echo ""
echo "Output directory: ${OVS_OUTPUT_DIR}"
echo "Log file: ${OVS_LOG_FILE}"

# Partial snapshots are not successful OVS runs.
if [[ "${ALL_COMPLETE}" != "true" ]]; then
    echo ""
    echo "Warning: Some samples are still missing after ${OVS_MAX_RETRY} retry attempts"
    echo "Consider increasing OVS_MAX_RETRY or checking the log for errors"
    exit 1
fi

exit ${OVS_EXIT_CODE}
