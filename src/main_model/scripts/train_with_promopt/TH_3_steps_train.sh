#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/home/leizy/24sum/CFRL/logs/with_prompt/T5_3_steps/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

cat > "${LOG_DIR}/run_info.txt" <<EOF
timestamp: ${TIMESTAMP}
script: $(basename "$0")
train_file: /home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt/3_steps/jsonl_data/train.json
validation_file: /home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt/3_steps/jsonl_data/val.json
output_dir: /home/leizy/24sum/CFRL/src/main_model/trained_model/with_prompt/trained_together/T5_3_steps
cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}
EOF

{
    python3 /home/leizy/24sum/CFRL/src/main_model/run/train.py \
        --model_name_or_path /home/leizy/24fall/src/main_model/T5-base \
        --do_train \
        --seed=88 \
        --save_total_limit=1 \
        --train_file /home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt/3_steps/jsonl_data/train.json \
        --validation_file /home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt/3_steps/jsonl_data/val.json \
        --output_dir /home/leizy/24sum/CFRL/src/main_model/trained_model/with_prompt/trained_together/T5_3_steps \
        --per_device_train_batch_size=16 \
        --overwrite_output_dir \
        --predict_with_generate
    EXIT_CODE=$?
    echo "exit_code: ${EXIT_CODE}" >> "${LOG_DIR}/run_info.txt"
    exit ${EXIT_CODE}
} 2>&1 | tee "${LOG_DIR}/train.log"