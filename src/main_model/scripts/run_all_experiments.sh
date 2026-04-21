#!/bin/bash

# One-click script to run all experiments
# This script will automatically run all training and testing for both with_prompt and no_prompt
# Usage: ./run_all_experiments.sh

set -e

echo "=========================================="
echo "Starting All CFRL Experiments"
echo "=========================================="
echo "This will run:"
echo "- with_prompt: 1-step, 3-step, 5-step predictions"
echo "- no_prompt: 1-step, 3-step, 5-step predictions"
echo "Total: 6 experiment configurations"
echo "=========================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_GPU_SCRIPT="$SCRIPT_DIR/multi_gpu_train_test.sh"
CALCULATE_TIME_SCRIPT="$SCRIPT_DIR/calculate_all_times.sh"

# Check if scripts exist
if [ ! -f "$MULTI_GPU_SCRIPT" ]; then
    echo "Error: Multi-GPU script not found at $MULTI_GPU_SCRIPT"
    exit 1
fi

if [ ! -f "$CALCULATE_TIME_SCRIPT" ]; then
    echo "Error: Calculate time script not found at $CALCULATE_TIME_SCRIPT"
    exit 1
fi

# Make scripts executable
chmod +x "$MULTI_GPU_SCRIPT"
chmod +x "$CALCULATE_TIME_SCRIPT"

# Function to run experiments for a data type
run_experiments_for_type() {
    local data_type=$1
    echo ""
    echo "=========================================="
    echo "Starting experiments for: $data_type"
    echo "=========================================="
    
    # Run 1-step experiments
    echo "[$(date)] Starting $data_type 1-step experiments..."
    "$MULTI_GPU_SCRIPT" "$data_type" "1"
    if [ $? -eq 0 ]; then
        echo "[$(date)] $data_type 1-step experiments completed successfully"
    else
        echo "[$(date)] ERROR: $data_type 1-step experiments failed"
        return 1
    fi
    
    # Run 3-step experiments
    echo "[$(date)] Starting $data_type 3-step experiments..."
    "$MULTI_GPU_SCRIPT" "$data_type" "3"
    if [ $? -eq 0 ]; then
        echo "[$(date)] $data_type 3-step experiments completed successfully"
    else
        echo "[$(date)] ERROR: $data_type 3-step experiments failed"
        return 1
    fi
    
    # Run 5-step experiments
    echo "[$(date)] Starting $data_type 5-step experiments..."
    "$MULTI_GPU_SCRIPT" "$data_type" "5"
    if [ $? -eq 0 ]; then
        echo "[$(date)] $data_type 5-step experiments completed successfully"
    else
        echo "[$(date)] ERROR: $data_type 5-step experiments failed"
        return 1
    fi
    
    # Calculate total time for this data type
    echo "[$(date)] Calculating total time for $data_type..."
    "$CALCULATE_TIME_SCRIPT" "$data_type"
    
    echo ""
    echo "=========================================="
    echo "$data_type experiments completed successfully!"
    echo "=========================================="
}

# Main execution
echo "[$(date)] Starting all experiments..."

# Run with_prompt experiments
run_experiments_for_type "with_prompt"
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: with_prompt experiments failed"
    exit 1
fi

# Run no_prompt experiments
run_experiments_for_type "no_prompt"
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: no_prompt experiments failed"
    exit 1
fi

# Generate final summary
echo ""
echo "=========================================="
echo "Generating Final Summary"
echo "=========================================="

LOG_DIR="/home/leizy/24sum/CFRL/logs"
FINAL_SUMMARY="$LOG_DIR/final_experiment_summary.txt"

cat > "$FINAL_SUMMARY" << EOF
Final Experiment Summary
=======================
Completed at: $(date)

Experiments Completed:
- with_prompt 1-step: ✓
- with_prompt 3-step: ✓
- with_prompt 5-step: ✓
- no_prompt 1-step: ✓
- no_prompt 3-step: ✓
- no_prompt 5-step: ✓

Results Location:
- Models: /home/leizy/24sum/CFRL/src/main_model/trained_model/
- Predictions: /home/leizy/24sum/CFRL/data/predicted_results/
- Logs: /home/leizy/24sum/CFRL/logs/
- Time summaries: /home/leizy/24sum/CFRL/logs/*/overall_time_summary.txt

Individual Step Summaries:
- with_prompt: /home/leizy/24sum/CFRL/logs/with_prompt/T5_*_step/time_summary.txt
- no_prompt: /home/leizy/24sum/CFRL/logs/no_prompt/T5_*_step/time_summary.txt

Evaluation Results:
- with_prompt: /home/leizy/24sum/CFRL/logs/with_prompt/T5_*_step/summary.json
- no_prompt: /home/leizy/24sum/CFRL/logs/no_prompt/T5_*_step/summary.json
EOF

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "[$(date)] All experiments finished"
echo ""
echo "Final summary saved to: $FINAL_SUMMARY"
echo ""
echo "You can now check:"
echo "1. Training logs: /home/leizy/24sum/CFRL/logs/"
echo "2. Trained models: /home/leizy/24sum/CFRL/src/main_model/trained_model/"
echo "3. Prediction results: /home/leizy/24sum/CFRL/data/predicted_results/"
echo "4. Time summaries: /home/leizy/24sum/CFRL/logs/*/overall_time_summary.txt"
echo ""
echo "==========================================" 