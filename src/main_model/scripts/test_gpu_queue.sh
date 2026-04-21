#!/bin/bash

# Test script to verify GPU queue management
# This simulates the main script behavior with proper GPU management

set -e

echo "=========================================="
echo "Testing GPU Queue Management"
echo "=========================================="

# Get all available GPUs
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

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
    echo "  [DEBUG] Marked GPU $gpu_id as busy with PID $pid"
}

# Function to mark GPU as free
mark_gpu_free() {
    local gpu_id=$1
    unset GPU_PROCESSES[$gpu_id]
    echo "  [DEBUG] Marked GPU $gpu_id as free"
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
            sleep 5
        fi
    done
    
    echo $gpu_id
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
}

# Simulate the main script behavior
echo "Simulating main script behavior with 8 tasks..."

for i in {1..8}; do
    echo ""
    echo "[$(date)] Processing task $i"
    show_gpu_status
    
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

echo ""
echo "Waiting for all tasks to complete..."
wait

echo "All tasks completed!"
echo "GPU queue management test completed successfully!" 