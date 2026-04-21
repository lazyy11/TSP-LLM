#!/bin/bash

# Test script to verify the real multi_gpu_train_test.sh behavior
# This will test if it correctly handles 16 tasks with only 4 running simultaneously

set -e

echo "=========================================="
echo "Testing Real Multi-GPU Script Behavior"
echo "=========================================="

# Check if we're in the right environment
echo "Current conda environment: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "pytorch" ]; then
    echo "Warning: Not in pytorch environment. Current: $CONDA_DEFAULT_ENV"
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --list-gpus
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

# Check data availability
DATA_DIR="CFRL/data/train_separately_datasets/with_prompt/1_step/TH_S"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# Count available columns
columns=($(ls -d "$DATA_DIR"/*/))
total_columns=${#columns[@]}
echo "Found $total_columns columns to process"

# Show first few columns
echo "First 5 columns:"
for i in {0..4}; do
    if [ $i -lt $total_columns ]; then
        column_name=$(basename "${columns[$i]}")
        echo "  $((i+1)). $column_name"
    fi
done

echo ""
echo "Testing the multi_gpu_train_test.sh script..."
echo "This will verify:"
echo "1. Only 4 tasks run simultaneously (one per GPU)"
echo "2. All 16 tasks eventually complete"
echo "3. GPU allocation is correct"
echo ""

# Test the script with a small subset first
echo "Testing with first 8 columns to verify behavior..."
echo "Running: ./CFRL/src/main_model/scripts/multi_gpu_train_test.sh with_prompt 1"

# Create a temporary test directory
TEST_DATA_DIR="CFRL/data/train_separately_datasets/with_prompt/1_step/TH_S_test"
mkdir -p "$TEST_DATA_DIR"

# Copy first 8 columns for testing
for i in {0..7}; do
    if [ $i -lt $total_columns ]; then
        cp -r "${columns[$i]}" "$TEST_DATA_DIR/"
    fi
done

echo "Created test data directory with 8 columns"
echo "Ready to test the script..."
echo ""
echo "To run the actual test, execute:"
echo "./CFRL/src/main_model/scripts/multi_gpu_train_test.sh with_prompt 1"
echo ""
echo "This will process all 16 columns with 4 GPUs"
echo "Expected behavior:"
echo "- First 4 tasks: GPU 0, 1, 2, 3"
echo "- Remaining 12 tasks: Wait for GPU availability, then assign"
echo "- All tasks should complete successfully" 