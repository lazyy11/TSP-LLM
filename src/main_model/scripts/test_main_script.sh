#!/bin/bash

# Simple test for the main script GPU allocation
# This simulates the main script behavior

set -e

echo "=========================================="
echo "Testing Main Script GPU Allocation"
echo "=========================================="

# Get all available GPUs
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

# Global variable to track last assigned GPU
LAST_ASSIGNED_GPU=-1

# Array to track GPU usage by process IDs
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

# Function to get next available GPU with round-robin allocation
get_next_gpu() {
    local start_gpu=$((LAST_ASSIGNED_GPU + 1))
    if [ $start_gpu -ge $GPUS ]; then
        start_gpu=0
    fi
    
    # First pass: check from start_gpu to end
    local gpu_id=$start_gpu
    while [ $gpu_id -lt $GPUS ]; do
        if is_gpu_available $gpu_id; then
            LAST_ASSIGNED_GPU=$gpu_id
            echo $gpu_id
            return
        fi
        gpu_id=$((gpu_id + 1))
    done
    
    # Second pass: check from 0 to start_gpu-1
    gpu_id=0
    while [ $gpu_id -lt $start_gpu ]; do
        if is_gpu_available $gpu_id; then
            LAST_ASSIGNED_GPU=$gpu_id
            echo $gpu_id
            return
        fi
        gpu_id=$((gpu_id + 1))
    done
    
    echo "no_available"
}

# Function to wait for GPU availability
wait_for_gpu() {
    local gpu_id="no_available"
    
    while [ "$gpu_id" = "no_available" ]; do
        gpu_id=$(get_next_gpu)
        if [ "$gpu_id" = "no_available" ]; then
            echo "[$(date)] All GPUs busy, waiting for available GPU..."
            sleep 5
        fi
    done
    
    echo $gpu_id
}

# Simulate the main script behavior
echo "Simulating main script behavior with 8 tasks..."

for i in {1..8}; do
    echo "[$(date)] Processing task $i"
    
    # Wait for an available GPU
    gpu_id=$(wait_for_gpu)
    
    echo "[$(date)] Assigning GPU $gpu_id to task $i"
    
    # Simulate starting a task (like training)
    echo "Starting simulated task $i on GPU $gpu_id..."
    sleep 3 &
    task_pid=$!
    mark_gpu_busy $gpu_id $task_pid
    
    echo "Task $i started on GPU $gpu_id with PID $task_pid"
    
    # Small delay to avoid race conditions
    sleep 1
done

echo "Waiting for all tasks to complete..."
wait

echo "All tasks completed!"
echo "GPU allocation test completed successfully!" 