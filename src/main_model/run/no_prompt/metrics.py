import numpy as np
from sklearn.metrics import r2_score
import re

def np_evaluate(gt_output, pred_output):
    # 添加数据有效性检查
    if len(gt_output) == 0 or len(pred_output) == 0:
        return np.nan, np.nan, np.nan
    
    # 确保数据形状一致
    gt_output = np.asarray(gt_output).flatten()
    pred_output = np.asarray(pred_output).flatten()
    
    # 手动计算R2
    # ss_total = np.sum((gt_output - np.mean(gt_output)) ** 2)
    # ss_residual = np.sum((gt_output - pred_output) ** 2)
    # r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan

    r2 = r2_score(gt_output, pred_output)
    
    mae = MAE(gt_output, pred_output)
    rmse = RMSE(gt_output, pred_output)

    return rmse, mae, r2

def MAE(gt_output, pred_output):
    # 移除mask，计算所有数据的MAE
    n = len(gt_output)
    if n == 0:
        return np.nan
    
    mae = np.mean(np.abs(gt_output - pred_output))
    return mae

def mape_smape(gt_output, pred_output):
    # 添加数据有效性检查
    n = len(gt_output)
    if n == 0:
        return np.nan, np.nan
    
    # 防止除以零
    epsilon = 1e-10
    gt_output = np.where(gt_output == 0, epsilon, gt_output)
    
    mape = np.mean(np.abs((gt_output - pred_output) / gt_output)) * 100
    smape = np.mean(np.abs(gt_output - pred_output) / ((np.abs(gt_output) + np.abs(pred_output)) / 2)) * 100

    return mape, smape

def RMSE(gt_output, pred_output):
    # 添加数据有效性检查
    n = len(gt_output)
    if n == 0:
        return np.nan
    
    mse = np.mean(np.square(gt_output - pred_output))
    return np.sqrt(mse)

def metric_with_missing_rate(gt_text, predicted_text, dataset):
    output_data = []
    gt_data = []
    missing_count = 0

    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    if len(gt_text) != len(predicted_text):
        raise ValueError(f"Length mismatch: gt_text has {len(gt_text)} lines, but predicted_text has {len(predicted_text)} lines.")

    for i in range(len(gt_text)):
        predicted_line = predicted_text[i]
        gt_line = gt_text[i]

        try:
            predicted_values = np.array([float(val) for val in number_pattern.findall(predicted_line)])
            gt_values = np.array([float(val) for val in number_pattern.findall(gt_line)])

            if len(predicted_values) != len(gt_values):
                raise ValueError(f"Mismatch in number of values at line {i}: predicted has {len(predicted_values)}, which is {predicted_values}, but gt has {len(gt_values)}, which is {gt_values}")

            output_data.extend(predicted_values)
            gt_data.extend(gt_values)
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            missing_count += 1

    # 添加数据有效性检查
    if len(output_data) == 0 or len(gt_data) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, missing_count / len(gt_text)

    output = np.array(output_data)
    gt_output = np.array(gt_data)
    
    rmse, mae, r2 = np_evaluate(gt_output, output)
    mape, smape = mape_smape(gt_output, output)
    missing_rate = missing_count / len(gt_text)

    return rmse, mae, mape, smape, r2, missing_rate