#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)

# 固定使用的 GPU，可根据需要修改
export CUDA_VISIBLE_DEVICES=0,1
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[INFO] Starting 1-step training..."
bash "${SCRIPT_DIR}/TH_1_step_train.sh"

echo "[INFO] Starting 3-step training..."
bash "${SCRIPT_DIR}/TH_3_steps_train.sh"

echo "[INFO] Starting 5-step training..."
bash "${SCRIPT_DIR}/TH_5_steps_train.sh"

echo "[INFO] All trainings finished."

