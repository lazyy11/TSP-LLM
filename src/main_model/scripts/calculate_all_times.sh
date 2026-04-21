#!/bin/bash

# Script to calculate total training time for all steps
# Usage: ./calculate_all_times.sh [with_prompt|no_prompt]

set -e

DATA_TYPE=${1:-"no_prompt"}  # with_prompt or no_prompt
LOG_DIR="/home/leizy/24sum/CFRL/logs"
LOG_TYPE_DIR="$LOG_DIR/$DATA_TYPE"

if [ ! -d "$LOG_TYPE_DIR" ]; then
    echo "Error: Log directory $LOG_TYPE_DIR does not exist"
    exit 1
fi

echo "Calculating total training time for $DATA_TYPE..."

# Initialize totals
total_train_time_all=0
total_test_time_all=0
total_time_all=0
total_models_all=0

# Process each step type
for step_dir in "$LOG_TYPE_DIR"/T5_*_step; do
    if [ -d "$step_dir" ]; then
        step_name=$(basename "$step_dir")
        echo "Processing $step_name..."
        
        # Check if time summary exists
        if [ -f "$step_dir/time_summary.txt" ]; then
            # Extract times from summary
            train_time=$(grep "Training time:" -A 2 "$step_dir/time_summary.txt" | grep "Total:" | awk '{print $2}')
            test_time=$(grep "Testing time:" -A 2 "$step_dir/time_summary.txt" | grep "Total:" | awk '{print $2}')
            total_time=$(grep "Total time:" -A 2 "$step_dir/time_summary.txt" | grep "Total:" | awk '{print $2}')
            models=$(grep "Successful models:" "$step_dir/time_summary.txt" | awk '{print $3}')
            
            if [ ! -z "$train_time" ] && [ ! -z "$test_time" ] && [ ! -z "$total_time" ] && [ ! -z "$models" ]; then
                total_train_time_all=$((total_train_time_all + train_time))
                total_test_time_all=$((total_test_time_all + test_time))
                total_time_all=$((total_time_all + total_time))
                total_models_all=$((total_models_all + models))
                
                echo "  $step_name: ${train_time}s training, ${test_time}s testing, ${total_time}s total, ${models} models"
            fi
        else
            echo "  Warning: No time summary found for $step_name"
        fi
    fi
done

# Convert to hours, minutes, seconds
train_hours=$((total_train_time_all / 3600))
train_minutes=$(((total_train_time_all % 3600) / 60))
train_seconds=$((total_train_time_all % 60))

test_hours=$((total_test_time_all / 3600))
test_minutes=$(((total_test_time_all % 3600) / 60))
test_seconds=$((total_test_time_all % 60))

total_hours=$((total_time_all / 3600))
total_minutes=$(((total_time_all % 3600) / 60))
total_seconds=$((total_time_all % 60))

# Save overall time summary
cat > "$LOG_TYPE_DIR/overall_time_summary.txt" << EOF
Overall Training Time Summary for $DATA_TYPE
============================================
Total successful models: $total_models_all

Training time:
  Total: ${total_train_time_all} seconds
  Formatted: ${train_hours}h ${train_minutes}m ${train_seconds}s

Testing time:
  Total: ${total_test_time_all} seconds
  Formatted: ${test_hours}h ${test_minutes}m ${test_seconds}s

Total time:
  Total: ${total_time_all} seconds
  Formatted: ${total_hours}h ${total_minutes}m ${total_seconds}s

Average time per model:
  Training: $((total_train_time_all / total_models_all)) seconds
  Testing: $((total_test_time_all / total_models_all)) seconds
  Total: $((total_time_all / total_models_all)) seconds

Breakdown by step:
EOF

# Add breakdown by step
for step_dir in "$LOG_TYPE_DIR"/T5_*_step; do
    if [ -d "$step_dir" ] && [ -f "$step_dir/time_summary.txt" ]; then
        step_name=$(basename "$step_dir")
        train_time=$(grep "Training time:" -A 2 "$step_dir/time_summary.txt" | grep "Total:" | awk '{print $2}')
        test_time=$(grep "Testing time:" -A 2 "$step_dir/time_summary.txt" | grep "Total:" | awk '{print $2}')
        total_time=$(grep "Total time:" -A 2 "$step_dir/time_summary.txt" | grep "Total:" | awk '{print $2}')
        models=$(grep "Successful models:" "$step_dir/time_summary.txt" | awk '{print $3}')
        
        if [ ! -z "$train_time" ] && [ ! -z "$test_time" ] && [ ! -z "$total_time" ] && [ ! -z "$models" ]; then
            echo "  $step_name: ${train_time}s training, ${test_time}s testing, ${total_time}s total, ${models} models" >> "$LOG_TYPE_DIR/overall_time_summary.txt"
        fi
    fi
done

echo "Overall time summary saved to $LOG_TYPE_DIR/overall_time_summary.txt"

# Print summary
echo ""
echo "Overall Summary for $DATA_TYPE:"
echo "Total models: $total_models_all"
echo "Total training time: ${train_hours}h ${train_minutes}m ${train_seconds}s"
echo "Total testing time: ${test_hours}h ${test_minutes}m ${test_seconds}s"
echo "Total time: ${total_hours}h ${total_minutes}m ${total_seconds}s"
echo "Average per model: $((total_time_all / total_models_all))s" 