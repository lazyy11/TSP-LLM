#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)

# 固定使用的 GPU，可根据需要修改
export CUDA_VISIBLE_DEVICES=1
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[INFO] Starting 1-step testing..."
bash "${SCRIPT_DIR}/TH_1_step_test.sh"

echo "[INFO] Starting 3-step testing..."
bash "${SCRIPT_DIR}/TH_3_steps_test.sh"

echo "[INFO] Starting 5-step testing..."
bash "${SCRIPT_DIR}/TH_5_steps_test.sh"

echo "[INFO] All testing finished."


