#!/bin/bash
# 临时训练脚本，只处理缺失的因子
set -e

# 临时替换数据目录
export DATA_DIR_OVERRIDE="/home/leizy/24sum/CFRL/logs/missing_factors_only_20250812_225840/temp_with_prompt_1"

# 运行原始训练脚本，但使用临时数据目录
bash "/home/leizy/24sum/CFRL/src/main_model/scripts/multi_gpu_train_test.sh" "with_prompt" "1"
