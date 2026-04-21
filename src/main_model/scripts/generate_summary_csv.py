#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import glob

def generate_summary_csv(data_type, step_type):
    """
    Generate a summary CSV file for all climate factors in a specific step folder
    
    Args:
        data_type: 'with_prompt' or 'no_prompt'
        step_type: '1', '3', or '5'
    """
    
    # Define paths
    base_pred_dir = f"/home/leizy/24sum/CFRL/data/train_together_results/predicted_results/{data_type}/TH/{step_type}_step"
    output_csv = f"/home/leizy/24sum/CFRL/data/train_together_results/predicted_results/{data_type}/TH/{step_type}_step_summary.csv"
    
    # Check if base directory exists
    if not os.path.exists(base_pred_dir):
        print(f"Error: Directory {base_pred_dir} does not exist")
        return
    
    # Get all climate factor directories
    climate_factors = []
    for item in os.listdir(base_pred_dir):
        item_path = os.path.join(base_pred_dir, item)
        if os.path.isdir(item_path):
            climate_factors.append(item)
    
    climate_factors.sort()  # Sort alphabetically
    
    print(f"Found {len(climate_factors)} climate factors in {base_pred_dir}")
    
    # Collect data for each climate factor
    summary_data = []
    
    for factor in climate_factors:
        metrics_file = os.path.join(base_pred_dir, factor, "metrics.json")
        
        if not os.path.exists(metrics_file):
            print(f"Warning: Metrics file not found for {factor}")
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Extract metrics
            metrics = data.get('metrics', {})
            missing_rate = data.get('missing_rate', 0)
            
            summary_data.append({
                'Variable': factor,
                'MAE': metrics.get('mae', float('nan')),
                'RMSE': metrics.get('rmse', float('nan')),
                'SMAPE (%)': metrics.get('smape', float('nan')),
                'R^2': metrics.get('r2', float('nan')),
                'Missing Rate (%)': missing_rate
            })
            
            print(f"Processed {factor}")
            
        except Exception as e:
            print(f"Error processing {factor}: {e}")
            continue
    
    if not summary_data:
        print("No valid data found")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    
    # Sort by Variable name
    df = df.sort_values('Variable')
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"Summary CSV saved to: {output_csv}")
    print(f"Total climate factors processed: {len(summary_data)}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Average MAE: {df['MAE'].mean():.6f}")
    print(f"Average RMSE: {df['RMSE'].mean():.6f}")
    print(f"Average SMAPE: {df['SMAPE (%)'].mean():.6f}%")
    print(f"Average R²: {df['R^2'].mean():.6f}")
    print(f"Average Missing Rate: {df['Missing Rate (%)'].mean():.6f}%")

def main():
    parser = argparse.ArgumentParser(description="Generate summary CSV for evaluation results")
    parser.add_argument("--data_type", required=True, type=str,
                       help="Data type: with_prompt or no_prompt or with_prompt_510")
    parser.add_argument("--step_type", required=True, choices=['1', '3', '5'], 
                       help="Step type: 1, 3, or 5")
    
    args = parser.parse_args()
    
    generate_summary_csv(args.data_type, args.step_type)

if __name__ == "__main__":
    main() 