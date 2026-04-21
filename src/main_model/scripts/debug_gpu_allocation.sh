#!/bin/bash

# Debug script to verify GPU round-robin allocation
# This will show exactly how GPUs are being assigned

set -e

echo "=========================================="
echo "Debugging GPU Round-Robin Allocation"
echo "=========================================="

# Get all available GPUs
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

# Global variable to track last assigned GPU
export LAST_ASSIGNED_GPU=-1

# Array to track GPU usage by process IDs
declare -gA GPU_PROCESSES

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
    echo "  [DEBUG] Marked GPU $gpu_id as busy with PID $pid"
}

# Function to mark GPU as free
mark_gpu_free() {
    local gpu_id=$1
    unset GPU_PROCESSES[$gpu_id]
    echo "  [DEBUG] Marked GPU $gpu_id as free"
}

# Function to get next available GPU with round-robin allocation
get_next_gpu() {
    # Try to find any available GPU, starting from the next one after last assigned
    local start_gpu=$((LAST_ASSIGNED_GPU + 1))
    if [ $start_gpu -ge $GPUS ]; then
        start_gpu=0
    fi
    
    echo "  [DEBUG] Looking for available GPU starting from $start_gpu"
    
    # First pass: check from start_gpu to end
    local gpu_id=$start_gpu
    while [ $gpu_id -lt $GPUS ]; do
        echo "  [DEBUG] Checking GPU $gpu_id"
        if is_gpu_available $gpu_id; then
            echo "  [DEBUG] GPU $gpu_id is available"
            LAST_ASSIGNED_GPU=$gpu_id
            echo $gpu_id
            return
        else
            echo "  [DEBUG] GPU $gpu_id is busy"
        fi
        gpu_id=$((gpu_id + 1))
    done
    
    # Second pass: check from 0 to start_gpu-1
    gpu_id=0
    while [ $gpu_id -lt $start_gpu ]; do
        echo "  [DEBUG] Checking GPU $gpu_id (second pass)"
        if is_gpu_available $gpu_id; then
            echo "  [DEBUG] GPU $gpu_id is available"
            LAST_ASSIGNED_GPU=$gpu_id
            echo $gpu_id
            return
        else
            echo "  [DEBUG] GPU $gpu_id is busy"
        fi
        gpu_id=$((gpu_id + 1))
    done
    
    echo "no_available"
}

# Function to show GPU status
show_gpu_status() {
    echo "  [DEBUG] Current GPU status:"
    for i in $(seq 0 $((GPUS-1))); do
        if is_gpu_available $i; then
            echo "    GPU $i: Available"
        else
            echo "    GPU $i: Busy (PID: ${GPU_PROCESSES[$i]})"
        fi
    done
    echo "  [DEBUG] LAST_ASSIGNED_GPU: $LAST_ASSIGNED_GPU"
}

# Test the allocation logic
echo "Testing GPU allocation for 8 tasks..."

for i in {1..8}; do
    echo ""
    echo "[$(date)] Processing task $i"
    show_gpu_status
    
    # Get next available GPU
    gpu_id=$(get_next_gpu)
    
    if [ "$gpu_id" != "no_available" ]; then
        echo "[$(date)] Assigning GPU $gpu_id to task $i"
        
        # Simulate starting a task
        echo "Starting simulated task $i on GPU $gpu_id..."
        sleep 2 &
        task_pid=$!
        mark_gpu_busy $gpu_id $task_pid
        
        echo "Task $i started on GPU $gpu_id with PID $task_pid"
    else
        echo "No GPU available for task $i"
    fi
    
    # Small delay
    sleep 1
done

echo ""
echo "Waiting for all tasks to complete..."
wait

echo "All tasks completed!"
echo "GPU allocation debug completed!" 