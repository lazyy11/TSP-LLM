#!/bin/bash

# Test script to verify GPU allocation
# Usage: ./test_gpu_allocation.sh

set -e

echo "=========================================="
echo "Testing GPU Allocation"
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

# Test GPU availability check
echo ""
echo "Testing GPU availability check:"
for i in $(seq 0 $((GPUS-1))); do
    if is_gpu_available $i; then
        echo "GPU $i: Available"
    else
        echo "GPU $i: Busy"
    fi
done

# Test round-robin allocation with simulated GPU usage
echo ""
echo "Testing round-robin allocation with simulated GPU usage:"
echo "Starting background processes to simulate GPU usage..."

# Start background processes to simulate GPU usage
for i in {0..2}; do
    if [ $i -lt $GPUS ]; then
        echo "Starting background process on GPU $i"
        CUDA_VISIBLE_DEVICES=$i bash -c "
echo 'Background process on GPU $i: Running...'
sleep 10
echo 'Background process on GPU $i: Finished'
" &
        process_pid=$!
        mark_gpu_busy $i $process_pid
        echo "Marked GPU $i as busy with PID $process_pid"
        sleep 1
    fi
done

# Wait a moment for processes to start
sleep 2

# Test allocation while some GPUs are busy
echo ""
echo "Testing allocation while some GPUs are busy:"
for i in {1..8}; do
    gpu_id=$(get_next_gpu)
    if [ "$gpu_id" != "no_available" ]; then
        echo "Task $i: Assigned to GPU $gpu_id"
        # Simulate starting a task on this GPU
        sleep 1 &
        task_pid=$!
        mark_gpu_busy $gpu_id $task_pid
        echo "  Started task on GPU $gpu_id with PID $task_pid"
    else
        echo "Task $i: No GPU available"
    fi
    sleep 1
done

echo "Waiting for background processes to complete..."
wait

# Clear all GPU processes
for i in $(seq 0 $((GPUS-1))); do
    mark_gpu_free $i
done

echo ""
echo "Testing allocation after all GPUs are free:"
for i in {1..4}; do
    gpu_id=$(get_next_gpu)
    if [ "$gpu_id" != "no_available" ]; then
        echo "Task $i: Assigned to GPU $gpu_id"
    else
        echo "Task $i: No GPU available"
    fi
done

echo ""
echo "GPU allocation test completed!"
echo "If you see tasks assigned to different GPUs in round-robin order, the allocation is working correctly." 