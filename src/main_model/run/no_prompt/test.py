import numpy as np

def read_values(file_path):
    values = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            cleaned = line.strip().rstrip('.')
            if not cleaned:
                print(f"警告：第{i+1}行为空")
                continue
                
            try:
                num = float(cleaned)
                # 添加数值范围检查
                if not (-1e6 < num < 1e6):
                    print(f"异常值警告：第{i+1}行数值超出正常范围: {num}")
            except ValueError as e:
                print(f"解析错误：第{i+1}行 '{line.strip()}' -> 清洗后 '{cleaned}'")
                continue
                
            values.append(num)
    return np.array(values)

def calculate_r2(gt_path, pred_path):
    y_true = read_values(gt_path)
    y_pred = read_values(pred_path)
    
    # 添加统计信息打印
    print(f"数据统计: 真实值共{len(y_true)}条，预测值共{len(y_pred)}条")
    print(f"真实值范围: [{np.min(y_true):.4f}, {np.max(y_true):.4f}]")
    print(f"预测值范围: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    # 新增残差分析
    residuals = y_true - y_pred
    top_indices = np.argsort(np.abs(residuals))[-5:][::-1]  # 取残差最大的5个点
    
    print("\n残差最大的5个数据点：")
    for idx in top_indices:
        print(f"第{idx+1}行: 真实值={y_true[idx]:.4f}, 预测值={y_pred[idx]:.4f}, 残差={residuals[idx]:.4f}")

    # 新增范围差异检查
    if (np.max(y_pred) / np.max(y_true)) > 10:
        print("\n警告：预测值量级显著大于真实值，请检查单位是否一致")

    # 添加极端值检查
    if np.any(np.abs(y_true - y_pred) > 1e4):
        print("警告：存在超过10000的残差")

    # 修改R²计算逻辑（处理零方差情况）
    ss_total = np.var(y_true) * len(y_true)
    if ss_total < 1e-9:  # 处理零方差情况
        print("警告：所有真实值相同（零方差）")
        return 0.0 if np.allclose(y_true, y_pred) else -np.inf
    
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return r2

# 使用示例
if __name__ == "__main__":
    r2 = calculate_r2("/home/leizy/24sum/CFRL/data/no_prompt/1_step/TH_S/humidity_land-wtd/test_y_prompt.txt", "/home/leizy/24sum/CFRL/data/predictied_results/no_prompt/TH/1_step/humidity_land-wtd/predicted.txt")
    print(f"R² Score: {r2:.4f}")