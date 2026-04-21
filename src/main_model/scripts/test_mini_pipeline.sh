#!/bin/bash

# Mini test script to verify the entire pipeline works
# This will test with minimal data: 1 climate factor, 1 step prediction

set -e

echo "=========================================="
echo "Starting Mini Pipeline Test"
echo "=========================================="

# Configuration
DATA_TYPE="with_prompt"
STEP_TYPE="1"
TEST_COLUMN="air-density_land-wtd"

# Paths
BASE_DATA_DIR="/home/leizy/24sum/CFRL/data/train_separately_datasets"
BASE_MODEL_DIR="/home/leizy/24sum/CFRL/src/main_model/trained_model"
BASE_RESULT_DIR="/home/leizy/24sum/CFRL/data/predicted_results"
LOG_DIR="/home/leizy/24sum/CFRL/logs/test_mini"

# Create test directories
mkdir -p "$LOG_DIR"

echo "[$(date)] Test configuration:"
echo "  Data type: $DATA_TYPE"
echo "  Step type: $STEP_TYPE"
echo "  Test column: $TEST_COLUMN"
echo "  Log directory: $LOG_DIR"

# Step 1: Check if data exists
DATA_PATH="$BASE_DATA_DIR/$DATA_TYPE/${STEP_TYPE}_step/TH_S/$TEST_COLUMN"
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data directory $DATA_PATH does not exist"
    exit 1
fi

echo "[$(date)] ✓ Data directory exists"

# Step 2: Check required files
for file in "train.json" "val.json" "test_x_prompt.txt" "test_y_prompt.txt"; do
    if [ ! -f "$DATA_PATH/$file" ]; then
        echo "Error: Required file $DATA_PATH/$file does not exist"
        exit 1
    fi
done

echo "[$(date)] ✓ All required data files exist"

# Step 3: Test training (with conda environment)
echo "[$(date)] Testing training..."
MODEL_DIR="$BASE_MODEL_DIR/$DATA_TYPE/T5_${STEP_TYPE}_step/$TEST_COLUMN"
mkdir -p "$MODEL_DIR"

# Activate conda environment and run training
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

echo "[$(date)] Running training with conda environment..."
CUDA_VISIBLE_DEVICES=0 python3 /home/leizy/24sum/CFRL/src/main_model/run/train.py \
    --model_name_or_path /home/leizy/24fall/src/main_model/T5-base \
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file "$DATA_PATH/train.json" \
    --validation_file "$DATA_PATH/val.json" \
    --output_dir "$MODEL_DIR" \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs=1 \
    --learning_rate=5e-5 \
    --warmup_steps=50 \
    --logging_steps=10 \
    --save_steps=100 \
    --max_steps=50 \
    2>&1 | tee "$LOG_DIR/training.log"

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Training completed successfully"
else
    echo "[$(date)] ✗ Training failed"
    exit 1
fi

# Step 4: Test testing
echo "[$(date)] Testing prediction..."
RESULT_DIR="$BASE_RESULT_DIR/$DATA_TYPE/TH/${STEP_TYPE}_step/$TEST_COLUMN"
mkdir -p "$RESULT_DIR"

echo "[$(date)] Running testing with conda environment..."
CUDA_VISIBLE_DEVICES=0 python3 /home/leizy/24sum/CFRL/src/main_model/run/test.py \
    -t "$DATA_PATH" \
    -m "$MODEL_DIR" \
    -s "$RESULT_DIR" \
    -d "$TEST_COLUMN" \
    --model_name T5 \
    -b 16 \
    2>&1 | tee "$LOG_DIR/testing.log"

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Testing completed successfully"
else
    echo "[$(date)] ✗ Testing failed"
    exit 1
fi

# Step 5: Test evaluation
echo "[$(date)] Testing evaluation..."
if [ -f "$RESULT_DIR/predicted.txt" ]; then
    echo "[$(date)] Running evaluation..."
    python3 /home/leizy/24sum/CFRL/src/main_model/run/evaluate.py \
        --gt_file "$DATA_PATH/test_y_prompt.txt" \
        --pred_file "$RESULT_DIR/predicted.txt" \
        --output_file "$RESULT_DIR/metrics.json" \
        --column_name "$TEST_COLUMN" \
        2>&1 | tee "$LOG_DIR/evaluation.log"
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] ✓ Evaluation completed successfully"
    else
        echo "[$(date)] ✗ Evaluation failed"
        exit 1
    fi
else
    echo "[$(date)] ✗ Prediction file not found"
    exit 1
fi

# Step 6: Test summary generation
echo "[$(date)] Testing summary generation..."
python3 /home/leizy/24sum/CFRL/src/main_model/scripts/generate_summary.py \
    --data_type "$DATA_TYPE" \
    --step_type "$STEP_TYPE" \
    --result_dir "$BASE_RESULT_DIR" \
    --output_file "$LOG_DIR/summary.json" \
    2>&1 | tee "$LOG_DIR/summary.log"

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Summary generation completed successfully"
else
    echo "[$(date)] ✗ Summary generation failed"
    exit 1
fi

# Step 7: Test CSV generation
echo "[$(date)] Testing CSV generation..."
python3 /home/leizy/24sum/CFRL/src/main_model/scripts/generate_summary_csv.py \
    --data_type "$DATA_TYPE" \
    --step_type "$STEP_TYPE" \
    2>&1 | tee "$LOG_DIR/csv.log"

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ CSV generation completed successfully"
else
    echo "[$(date)] ✗ CSV generation failed"
    exit 1
fi

# Step 8: Display results
echo ""
echo "=========================================="
echo "Mini Pipeline Test Results"
echo "=========================================="
echo "[$(date)] All tests completed successfully!"

echo ""
echo "Generated files:"
echo "  Model: $MODEL_DIR"
echo "  Predictions: $RESULT_DIR/predicted.txt"
echo "  Metrics: $RESULT_DIR/metrics.json"
echo "  Summary: $LOG_DIR/summary.json"
echo "  CSV: $BASE_RESULT_DIR/$DATA_TYPE/TH/${STEP_TYPE}_step_summary.csv"

echo ""
echo "Log files:"
echo "  Training: $LOG_DIR/training.log"
echo "  Testing: $LOG_DIR/testing.log"
echo "  Evaluation: $LOG_DIR/evaluation.log"
echo "  Summary: $LOG_DIR/summary.log"
echo "  CSV: $LOG_DIR/csv.log"

echo ""
echo "=========================================="
echo "Mini Pipeline Test PASSED! ✓"
echo "==========================================" 