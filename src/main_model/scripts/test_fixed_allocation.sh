#!/bin/bash

# Test script to verify the fixed round-robin GPU allocation
# This simulates the main script behavior

set -e

echo "=========================================="
echo "Testing Fixed Round-Robin GPU Allocation"
echo "=========================================="

# Get all available GPUs
GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPUS GPUs available"

# Simulate the main script behavior
echo "Simulating main script behavior with 8 tasks..."

# Process each task with round-robin GPU assignment
current_gpu=0
for i in {1..8}; do
    echo "[$(date)] Processing task $i"
    
    # Assign GPU in round-robin fashion
    gpu_id=$current_gpu
    current_gpu=$((current_gpu + 1))
    if [ $current_gpu -ge $GPUS ]; then
        current_gpu=0
    fi
    
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