#!/bin/bash
# evaluate_predictions.sh
# 对已有的预测结果进行评估并生成汇总报告

set -euo pipefail

# ============ 配置参数 ============
# 从命令行参数读取，如果没有则使用默认值
DATA_TYPE=${1:-"with_prompt"}  # with_prompt 或 no_prompt
STEP_TYPE=${2:-"1"}             # 1, 3, 或 5

# 根据 STEP_TYPE 确定正确的命名（1用单数，3和5用复数）
if [ "$STEP_TYPE" = "1" ]; then
    STEP_NAME="1_step"
    MODEL_STEP="1_step"
elif [ "$STEP_TYPE" = "3" ]; then
    STEP_NAME="3_steps"
    MODEL_STEP="3_steps"
elif [ "$STEP_TYPE" = "5" ]; then
    STEP_NAME="5_steps"
    MODEL_STEP="5_steps"
else
    echo "Error: Invalid STEP_TYPE. Must be 1, 3, or 5"
    exit 1
fi

# 路径配置
BASE_DATA_DIR="/home/leizy/24sum/CFRL/data/train_together_datasets"
BASE_RESULT_DIR="/home/leizy/24sum/CFRL/data/train_together_results/predicted_results"
LOG_DIR="/home/leizy/24sum/CFRL/logs/evaluation"

# 创建日志目录
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${DATA_TYPE}_${STEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
echo "Data type: $DATA_TYPE"
echo "Step type: $STEP_TYPE (${STEP_NAME})"
echo "Log file: $LOG_FILE"
echo ""

# 激活conda环境（如果需要）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pytorch

# ============ 评估所有预测结果 ============
DATA_DIR="$BASE_DATA_DIR/$DATA_TYPE/${STEP_NAME}/TH_S"
RESULT_DIR="$BASE_RESULT_DIR/$DATA_TYPE/TH/${STEP_NAME}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist" | tee -a "$LOG_FILE"
    exit 1
fi

# 统计变量
total_count=0
success_count=0
failed_count=0

echo "[$(date)] Processing all datasets..." | tee -a "$LOG_FILE"

for DIR in "$DATA_DIR"/*/; do
    DIR_NAME=$(basename "$DIR")
    GT_FILE="$DIR/test_y_prompt.txt"
    PRED_FILE="$RESULT_DIR/$DIR_NAME/predicted.txt"
    OUTPUT_FILE="$RESULT_DIR/$DIR_NAME/metrics.json"
    
    total_count=$((total_count + 1))
    
    # 检查文件是否存在
    if [ ! -f "$GT_FILE" ]; then
        echo "[$(date)] Warning: Ground truth file not found for $DIR_NAME" | tee -a "$LOG_FILE"
        failed_count=$((failed_count + 1))
        continue
    fi
    
    if [ ! -f "$PRED_FILE" ]; then
        echo "[$(date)] Warning: Prediction file not found for $DIR_NAME" | tee -a "$LOG_FILE"
        failed_count=$((failed_count + 1))
        continue
    fi
    
    # 创建输出目录
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    
    # 运行评估
    echo "[$(date)] Evaluating $DIR_NAME..." | tee -a "$LOG_FILE"
    
    python3 /home/leizy/24sum/CFRL/src/main_model/run/evaluate.py \
        --gt_file "$GT_FILE" \
        --pred_file "$PRED_FILE" \
        --output_file "$OUTPUT_FILE" \
        --column_name "$DIR_NAME" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] ✓ Evaluation completed for $DIR_NAME" | tee -a "$LOG_FILE"
        success_count=$((success_count + 1))
    else
        echo "[$(date)] ✗ Evaluation failed for $DIR_NAME" | tee -a "$LOG_FILE"
        failed_count=$((failed_count + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=========================================="| tee -a "$LOG_FILE"
echo "Individual Evaluation Summary" | tee -a "$LOG_FILE"
echo "=========================================="| tee -a "$LOG_FILE"
echo "Total datasets: $total_count" | tee -a "$LOG_FILE"
echo "Successful: $success_count" | tee -a "$LOG_FILE"
echo "Failed: $failed_count" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============ 生成汇总报告 ============
if [ $success_count -gt 0 ]; then
    echo "[$(date)] Generating summary report..." | tee -a "$LOG_FILE"
    
    python3 /home/leizy/24sum/CFRL/src/main_model/scripts/generate_summary.py \
        --data_type "$DATA_TYPE" \
        --step_type "$STEP_TYPE" \
        --result_dir "$BASE_RESULT_DIR" \
        --output_file "$LOG_DIR/summary_${DATA_TYPE}_${STEP_NAME}.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "[$(date)] Generating summary CSV..." | tee -a "$LOG_FILE"
    
    python3 /home/leizy/24sum/CFRL/src/main_model/scripts/generate_summary_csv.py \
        --data_type "$DATA_TYPE" \
        --step_type "$STEP_TYPE" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "=========================================="| tee -a "$LOG_FILE"
    echo "Evaluation Completed Successfully!" | tee -a "$LOG_FILE"
    echo "=========================================="| tee -a "$LOG_FILE"
    echo "Results:" | tee -a "$LOG_FILE"
    echo "  Individual metrics: $RESULT_DIR/*/metrics.json" | tee -a "$LOG_FILE"
    echo "  Summary JSON: $LOG_DIR/summary_${DATA_TYPE}_${STEP_NAME}.json" | tee -a "$LOG_FILE"
    echo "  Summary CSV: $BASE_RESULT_DIR/$DATA_TYPE/TH/${STEP_NAME}_summary.csv" | tee -a "$LOG_FILE"
    echo "  Log file: $LOG_FILE" | tee -a "$LOG_FILE"
else
    echo "[$(date)] No successful evaluations, skipping summary generation" | tee -a "$LOG_FILE"
fi

