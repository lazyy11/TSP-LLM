#!/bin/bash

# Multi-GPU training and testing script for CFRL
# Supports both with_prompt and no_prompt datasets
# Usage: ./multi_gpu_train_test.sh [with_prompt|no_prompt] [1|3|5]

set -e

# Configuration
DATA_TYPE=${1:-"no_prompt"}  # with_prompt or no_prompt
STEP_TYPE=${2:-"1"}          # 1, 3, or 5
BASE_DATA_DIR="/home/leizy/24sum/CFRL/data/train_separately_datasets"
BASE_MODEL_DIR="/home/leizy/24sum/CFRL/src/main_model/trained_model"
BASE_RESULT_DIR="/home/leizy/24sum/CFRL/data/predicted_results"
LOG_DIR="/home/leizy/24sum/CFRL/logs"
T5_MODEL_PATH="/home/leizy/24fall/src/main_model/T5-base"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$BASE_RESULT_DIR"

# Create log directory structure
LOG_TYPE_DIR="$LOG_DIR/$DATA_TYPE"
LOG_STEP_DIR="$LOG_TYPE_DIR/T5_${STEP_TYPE}_step"
mkdir -p "$LOG_STEP_DIR"

# Get all available GPUs
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

# GPU queue management
declare -A GPU_PROCESSES

# Function to check if GPU is available
is_gpu_available() {
    local gpu_id=$1
    
    # Check if we have a process ID for this GPU
    if [ -n "${GPU_PROCESSES[$gpu_id]}" ]; then
        # Check if the process is still running
        if kill -0 "${GPU_PROCESSES[$gpu_id]}" 2>/dev/null; then
            return 1  # GPU is busy
        else
            # Process is dead, clear it
            unset GPU_PROCESSES[$gpu_id]
        fi
    fi
    
    return 0  # GPU is available
}

# Function to mark GPU as busy
mark_gpu_busy() {
    local gpu_id=$1
    local pid=$2
    GPU_PROCESSES[$gpu_id]=$pid
}

# Function to mark GPU as free
mark_gpu_free() {
    local gpu_id=$1
    unset GPU_PROCESSES[$gpu_id]
}

# Function to get next available GPU
get_next_available_gpu() {
    for gpu_id in $(seq 0 $((GPUS-1))); do
        if is_gpu_available $gpu_id; then
            echo $gpu_id
            return
        fi
    done
    echo "no_available"
}

# Function to wait for GPU availability
wait_for_gpu() {
    local gpu_id="no_available"
    
    while [ "$gpu_id" = "no_available" ]; do
        gpu_id=$(get_next_available_gpu)
        if [ "$gpu_id" = "no_available" ]; then
            echo "[$(date)] All GPUs busy, waiting for available GPU..."
            sleep 10
        fi
    done
    
    echo $gpu_id
}

# Function to train and test a single model
train_and_test() {
    local gpu_id=$1
    local data_path=$2
    local column_name=$3
    local data_type=$4
    local step_type=$5
    
    local model_dir="$BASE_MODEL_DIR/$data_type/T5_${step_type}_step/$column_name"
    local result_dir="$BASE_RESULT_DIR/$data_type/TH/${step_type}_step/$column_name"
    local log_file="$LOG_STEP_DIR/${column_name}_gpu${gpu_id}.log"
    local time_file="$LOG_STEP_DIR/${column_name}_gpu${gpu_id}_time.txt"
    
    echo "[$(date)] Starting training for $column_name on GPU $gpu_id" | tee -a "$log_file"
    
    # Create directories
    mkdir -p "$model_dir"
    mkdir -p "$result_dir"
    
    # Record start time
    start_time=$(date +%s)
    echo "Start time: $(date)" > "$time_file"
    
    # Training
    echo "[$(date)] Training model for $column_name..." | tee -a "$log_file"
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate pytorch
    CUDA_VISIBLE_DEVICES=$gpu_id python3 /home/leizy/24sum/CFRL/src/main_model/run/train.py \
        --model_name_or_path "$T5_MODEL_PATH" \
        --do_train \
        --seed=88 \
        --save_total_limit=1 \
        --train_file "$data_path/train.json" \
        --validation_file "$data_path/val.json" \
        --output_dir "$model_dir" \
        --per_device_train_batch_size=16 \
        --overwrite_output_dir \
        --predict_with_generate \
        --num_train_epochs=10 \
        --learning_rate=5e-5 \
        --warmup_steps=500 \
        --logging_steps=100 \
        --save_steps=1000 \
        2>&1 | tee -a "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] Training completed successfully for $column_name" | tee -a "$log_file"
        
        # Record training end time
        train_end_time=$(date +%s)
        train_duration=$((train_end_time - start_time))
        echo "Training end time: $(date)" >> "$time_file"
        echo "Training duration: ${train_duration} seconds" >> "$time_file"
        
        # Testing
        echo "[$(date)] Starting testing for $column_name..." | tee -a "$log_file"
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate pytorch
        CUDA_VISIBLE_DEVICES=$gpu_id python3 /home/leizy/24sum/CFRL/src/main_model/run/test.py \
            -t "$data_path" \
            -m "$model_dir" \
            -s "$result_dir" \
            -d "$column_name" \
            --model_name T5 \
            -b 16 \
            2>&1 | tee -a "$log_file"
        
        if [ $? -eq 0 ]; then
            echo "[$(date)] Testing completed successfully for $column_name" | tee -a "$log_file"
            
            # Evaluation
            echo "[$(date)] Starting evaluation for $column_name..." | tee -a "$log_file"
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate pytorch
            python3 /home/leizy/24sum/CFRL/src/main_model/run/evaluate.py \
                --gt_file "$data_path/test_y_prompt.txt" \
                --pred_file "$result_dir/predicted.txt" \
                --output_file "$result_dir/metrics.json" \
                --column_name "$column_name" \
                2>&1 | tee -a "$log_file"
            
            # Record total end time
            end_time=$(date +%s)
            total_duration=$((end_time - start_time))
            echo "Total end time: $(date)" >> "$time_file"
            echo "Total duration: ${total_duration} seconds" >> "$time_file"
            echo "Testing duration: $((total_duration - train_duration)) seconds" >> "$time_file"
            
            echo "[$(date)] All tasks completed successfully for $column_name" | tee -a "$log_file"
        else
            echo "[$(date)] Testing failed for $column_name" | tee -a "$log_file"
        fi
    else
        echo "[$(date)] Training failed for $column_name" | tee -a "$log_file"
    fi
    
    # Mark GPU as free when task completes
    mark_gpu_free $gpu_id
    echo "[$(date)] Task completed for $column_name on GPU $gpu_id"
}

# Function to calculate total training time
calculate_total_time() {
    local log_step_dir="$1"
    local total_train_time=0
    local total_test_time=0
    local total_time=0
    local successful_models=0
    
    echo "Calculating total training time for $log_step_dir..."
    
    for time_file in "$log_step_dir"/*_time.txt; do
        if [ -f "$time_file" ]; then
            train_duration=$(grep "Training duration:" "$time_file" | awk '{print $3}')
            total_duration=$(grep "Total duration:" "$time_file" | awk '{print $3}')
            
            if [ ! -z "$train_duration" ] && [ ! -z "$total_duration" ]; then
                total_train_time=$((total_train_time + train_duration))
                total_test_time=$((total_test_time + total_duration - train_duration))
                total_time=$((total_time + total_duration))
                successful_models=$((successful_models + 1))
            fi
        fi
    done
    
    # Convert to hours, minutes, seconds
    train_hours=$((total_train_time / 3600))
    train_minutes=$(((total_train_time % 3600) / 60))
    train_seconds=$((total_train_time % 60))
    
    test_hours=$((total_test_time / 3600))
    test_minutes=$(((total_test_time % 3600) / 60))
    test_seconds=$((total_test_time % 60))
    
    total_hours=$((total_time / 3600))
    total_minutes=$(((total_time % 3600) / 60))
    total_seconds=$((total_time % 60))
    
    # Save time summary
    cat > "$log_step_dir/time_summary.txt" << EOF
Training Time Summary for $DATA_TYPE T5_${STEP_TYPE}_step
====================================================
Successful models: $successful_models

Training time:
  Total: ${total_train_time} seconds
  Formatted: ${train_hours}h ${train_minutes}m ${train_seconds}s

Testing time:
  Total: ${total_test_time} seconds
  Formatted: ${test_hours}h ${test_minutes}m ${test_seconds}s

Total time:
  Total: ${total_time} seconds
  Formatted: ${total_hours}h ${total_minutes}m ${total_seconds}s

Average time per model:
  Training: $((total_train_time / successful_models)) seconds
  Testing: $((total_test_time / successful_models)) seconds
  Total: $((total_time / successful_models)) seconds
EOF
    
    echo "Time summary saved to $log_step_dir/time_summary.txt"
}

# Main execution
echo "[$(date)] Starting multi-GPU training and testing for $DATA_TYPE, ${STEP_TYPE}_step"

# Get all column directories
DATA_DIR="$BASE_DATA_DIR/$DATA_TYPE/${STEP_TYPE}_step/TH_S"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# Get all column directories
columns=($(ls -d "$DATA_DIR"/*/))
total_columns=${#columns[@]}
echo "Found $total_columns columns to process"

# Process each column with GPU queue management
for column_path in "${columns[@]}"; do
    column_name=$(basename "$column_path")
    echo "[$(date)] Processing column: $column_name"
    
    # Wait for an available GPU
    gpu_id=$(wait_for_gpu)
    
    echo "[$(date)] Assigning GPU $gpu_id to $column_name"
    
    # Start training and testing in background
    train_and_test "$gpu_id" "$column_path" "$column_name" "$DATA_TYPE" "$STEP_TYPE" &
    train_pid=$!
    
    # Mark GPU as busy
    mark_gpu_busy $gpu_id $train_pid
    
    # Small delay to avoid race conditions
    sleep 5
done

# Wait for all background processes to complete
echo "[$(date)] Waiting for all tasks to complete..."
wait

echo "[$(date)] All training and testing tasks completed!"

# Calculate total training time
calculate_total_time "$LOG_STEP_DIR"

# Generate summary report
echo "[$(date)] Generating summary report..."
python3 /home/leizy/24sum/CFRL/src/main_model/scripts/generate_summary.py \
    --data_type "$DATA_TYPE" \
    --step_type "$STEP_TYPE" \
    --result_dir "$BASE_RESULT_DIR" \
    --output_file "$LOG_STEP_DIR/summary.json"

# Generate summary CSV
echo "[$(date)] Generating summary CSV..."
python3 /home/leizy/24sum/CFRL/src/main_model/scripts/generate_summary_csv.py \
    --data_type "$DATA_TYPE" \
    --step_type "$STEP_TYPE"

echo "[$(date)] Multi-GPU training and testing completed successfully!" 