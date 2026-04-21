#!/usr/bin/env bash
# evaluate_all_steps.sh
# 一次性评估所有步骤（1_step, 3_steps, 5_steps）

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 配置
DATA_TYPE=${1:-"with_prompt"}  # with_prompt 或 no_prompt

echo "=========================================="
echo "Evaluating All Steps: $DATA_TYPE"
echo "=========================================="
echo ""

# 评估 1_step
echo "[$(date)] Evaluating 1_step predictions..."
bash "$SCRIPT_DIR/evaluate_predictions.sh" "$DATA_TYPE" "1"
echo ""

# 评估 3_steps
echo "[$(date)] Evaluating 3_steps predictions..."
bash "$SCRIPT_DIR/evaluate_predictions.sh" "$DATA_TYPE" "3"
echo ""

# 评估 5_steps
echo "[$(date)] Evaluating 5_steps predictions..."
bash "$SCRIPT_DIR/evaluate_predictions.sh" "$DATA_TYPE" "5"
echo ""

echo "=========================================="
echo "All Evaluations Completed!"
echo "=========================================="

