#!/usr/bin/env python3

import argparse
import json
import numpy as np
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def extract_numbers(text):
    """Extract numbers from text using regex"""
    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    return [float(val) for val in number_pattern.findall(text)]

def calculate_metrics(gt_values, pred_values):
    """Calculate various metrics"""
    if len(gt_values) == 0 or len(pred_values) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "smape": np.nan
        }
    
    gt_values = np.array(gt_values)
    pred_values = np.array(pred_values)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(gt_values, pred_values))
    mae = mean_absolute_error(gt_values, pred_values)
    r2 = r2_score(gt_values, pred_values)
    
    # MAPE and SMAPE
    epsilon = 1e-10
    gt_values_safe = np.where(gt_values == 0, epsilon, gt_values)
    
    mape = np.mean(np.abs((gt_values - pred_values) / gt_values_safe)) * 100
    smape = np.mean(np.abs(gt_values - pred_values) / ((np.abs(gt_values) + np.abs(pred_values)) / 2)) * 100
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction results")
    parser.add_argument("--gt_file", required=True, help="Ground truth file path")
    parser.add_argument("--pred_file", required=True, help="Prediction file path")
    parser.add_argument("--output_file", required=True, help="Output metrics file path")
    parser.add_argument("--column_name", required=True, help="Column name for identification")
    
    args = parser.parse_args()
    
    # Read ground truth and prediction files
    with open(args.gt_file, 'r') as f:
        gt_lines = f.readlines()
    
    with open(args.pred_file, 'r') as f:
        pred_lines = f.readlines()
    
    # Extract all numbers from both files
    gt_values = []
    pred_values = []
    missing_count = 0
    
    for i, (gt_line, pred_line) in enumerate(zip(gt_lines, pred_lines)):
        try:
            gt_nums = extract_numbers(gt_line.strip())
            pred_nums = extract_numbers(pred_line.strip())
            
            if len(gt_nums) != len(pred_nums):
                print(f"Warning: Line {i+1} has different number of values. GT: {len(gt_nums)}, Pred: {len(pred_nums)}")
                missing_count += 1
                continue
            
            gt_values.extend(gt_nums)
            pred_values.extend(pred_nums)
            
        except Exception as e:
            print(f"Error processing line {i+1}: {e}")
            missing_count += 1
    
    # Calculate metrics
    metrics = calculate_metrics(gt_values, pred_values)
    
    # Add additional information
    result = {
        "column_name": args.column_name,
        "total_samples": len(gt_lines),
        "valid_samples": len(gt_lines) - missing_count,
        "missing_rate": (missing_count / len(gt_lines) * 100) if len(gt_lines) > 0 else 0,
        "metrics": metrics
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Evaluation completed for {args.column_name}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"MAPE: {metrics['mape']:.6f}%")
    print(f"SMAPE: {metrics['smape']:.6f}%")

if __name__ == "__main__":
    main()