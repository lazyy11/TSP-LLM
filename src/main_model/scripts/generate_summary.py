#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

def load_metrics(result_dir, data_type, step_type):
    """Load metrics from all column directories"""
    metrics_data = []
    base_path = f"{result_dir}/{data_type}/TH/{step_type}_step"
    
    if not os.path.exists(base_path):
        print(f"Warning: Path {base_path} does not exist")
        return metrics_data
    
    for column_dir in os.listdir(base_path):
        column_path = os.path.join(base_path, column_dir)
        if os.path.isdir(column_path):
            metrics_file = os.path.join(column_path, "metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    metrics_data.append(metrics)
                except Exception as e:
                    print(f"Error loading metrics for {column_dir}: {e}")
    
    return metrics_data

def calculate_summary_statistics(metrics_data):
    """Calculate summary statistics across all columns"""
    if not metrics_data:
        return {}
    
    # Extract all metric values
    rmse_values = [m['metrics']['rmse'] for m in metrics_data if not pd.isna(m['metrics']['rmse'])]
    mae_values = [m['metrics']['mae'] for m in metrics_data if not pd.isna(m['metrics']['mae'])]
    r2_values = [m['metrics']['r2'] for m in metrics_data if not pd.isna(m['metrics']['r2'])]
    mape_values = [m['metrics']['mape'] for m in metrics_data if not pd.isna(m['metrics']['mape'])]
    smape_values = [m['metrics']['smape'] for m in metrics_data if not pd.isna(m['metrics']['smape'])]
    missing_rates = [m['missing_rate'] for m in metrics_data]
    
    summary = {
        "total_columns": len(metrics_data),
        "successful_columns": len([m for m in metrics_data if not pd.isna(m['metrics']['rmse'])]),
        "metrics_summary": {
            "rmse": {
                "mean": float(np.mean(rmse_values)) if rmse_values else np.nan,
                "std": float(np.std(rmse_values)) if rmse_values else np.nan,
                "min": float(np.min(rmse_values)) if rmse_values else np.nan,
                "max": float(np.max(rmse_values)) if rmse_values else np.nan
            },
            "mae": {
                "mean": float(np.mean(mae_values)) if mae_values else np.nan,
                "std": float(np.std(mae_values)) if mae_values else np.nan,
                "min": float(np.min(mae_values)) if mae_values else np.nan,
                "max": float(np.max(mae_values)) if mae_values else np.nan
            },
            "r2": {
                "mean": float(np.mean(r2_values)) if r2_values else np.nan,
                "std": float(np.std(r2_values)) if r2_values else np.nan,
                "min": float(np.min(r2_values)) if r2_values else np.nan,
                "max": float(np.max(r2_values)) if r2_values else np.nan
            },
            "mape": {
                "mean": float(np.mean(mape_values)) if mape_values else np.nan,
                "std": float(np.std(mape_values)) if mape_values else np.nan,
                "min": float(np.min(mape_values)) if mape_values else np.nan,
                "max": float(np.max(mape_values)) if mape_values else np.nan
            },
            "smape": {
                "mean": float(np.mean(smape_values)) if smape_values else np.nan,
                "std": float(np.std(smape_values)) if smape_values else np.nan,
                "min": float(np.min(smape_values)) if smape_values else np.nan,
                "max": float(np.max(smape_values)) if smape_values else np.nan
            }
        },
        "missing_rate_summary": {
            "mean": float(np.mean(missing_rates)) if missing_rates else np.nan,
            "std": float(np.std(missing_rates)) if missing_rates else np.nan,
            "min": float(np.min(missing_rates)) if missing_rates else np.nan,
            "max": float(np.max(missing_rates)) if missing_rates else np.nan
        }
    }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate summary report for training and testing results")
    parser.add_argument("--data_type", required=True, type=str, help="Data type (e.g., with_prompt, no_prompt, with_prompt_510)")
    parser.add_argument("--step_type", required=True, choices=["1", "3", "5"], help="Step type")
    parser.add_argument("--result_dir", required=True, help="Base result directory")
    parser.add_argument("--output_file", required=True, help="Output summary file path")
    
    args = parser.parse_args()
    
    # Load metrics data
    print(f"Loading metrics for {args.data_type}, {args.step_type}_step...")
    metrics_data = load_metrics(args.result_dir, args.data_type, args.step_type)
    
    if not metrics_data:
        print("No metrics data found!")
        return
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(metrics_data)
    
    # Add individual column results
    summary["individual_results"] = metrics_data
    
    # Add metadata
    summary["metadata"] = {
        "data_type": args.data_type,
        "step_type": args.step_type,
        "result_dir": args.result_dir
    }
    
    # Save summary
    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\nSummary Report for {args.data_type}, {args.step_type}_step:")
    print(f"Total columns processed: {summary['total_columns']}")
    print(f"Successful columns: {summary['successful_columns']}")
    
    if summary['metrics_summary']['rmse']['mean'] is not None:
        print(f"\nRMSE - Mean: {summary['metrics_summary']['rmse']['mean']:.6f}, Std: {summary['metrics_summary']['rmse']['std']:.6f}")
        print(f"MAE - Mean: {summary['metrics_summary']['mae']['mean']:.6f}, Std: {summary['metrics_summary']['mae']['std']:.6f}")
        print(f"R² - Mean: {summary['metrics_summary']['r2']['mean']:.6f}, Std: {summary['metrics_summary']['r2']['std']:.6f}")
        print(f"MAPE - Mean: {summary['metrics_summary']['mape']['mean']:.6f}%, Std: {summary['metrics_summary']['mape']['std']:.6f}%")
        print(f"SMAPE - Mean: {summary['metrics_summary']['smape']['mean']:.6f}%, Std: {summary['metrics_summary']['smape']['std']:.6f}%")
    
    print(f"\nSummary saved to: {args.output_file}")

if __name__ == "__main__":
    import numpy as np
    main() 