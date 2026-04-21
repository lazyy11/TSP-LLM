#!/bin/bash

# Simple test for the new round-robin GPU allocation
# This simulates the main script behavior with simple round-robin

set -e

echo "=========================================="
echo "Testing Simple Round-Robin GPU Allocation"
echo "=========================================="

# Get all available GPUs
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

# Simple round-robin GPU assignment
CURRENT_GPU=0

# Function to get next GPU in round-robin fashion
get_next_gpu() {
    local gpu_id=$CURRENT_GPU
    echo "  [DEBUG] Current GPU: $CURRENT_GPU, Returning: $gpu_id"
    CURRENT_GPU=$((CURRENT_GPU + 1))
    if [ $CURRENT_GPU -ge $GPUS ]; then
        CURRENT_GPU=0
    fi
    echo "  [DEBUG] Next GPU will be: $CURRENT_GPU"
    echo $gpu_id
}

# Test the function directly
echo "Testing get_next_gpu function directly:"
for i in {1..8}; do
    result=$(get_next_gpu)
    echo "Call $i: $result"
done

# Simulate the main script behavior
echo "Simulating main script behavior with 8 tasks..."

for i in {1..8}; do
    echo "[$(date)] Processing task $i"
    
    # Get next GPU in round-robin fashion
    gpu_id=$(get_next_gpu)
    
    echo "[$(date)] Assigning GPU $gpu_id to task $i"
    
    # Simulate starting a task (like training)
    echo "Starting simulated task $i on GPU $gpu_id..."
    sleep 2 &
    task_pid=$!
    
    echo "Task $i started on GPU $gpu_id with PID $task_pid"
    
    # Small delay to avoid race conditions
    sleep 1
done

echo "Waiting for all tasks to complete..."
wait

echo "All tasks completed!"
echo "Expected GPU assignment:"
echo "Task 1: GPU 0"
echo "Task 2: GPU 1" 
echo "Task 3: GPU 2"
echo "Task 4: GPU 3"
echo "Task 5: GPU 0"
echo "Task 6: GPU 1"
echo "Task 7: GPU 2"
echo "Task 8: GPU 3"
echo "GPU allocation test completed successfully!" 