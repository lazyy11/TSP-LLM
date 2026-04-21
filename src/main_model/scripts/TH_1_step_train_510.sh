#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python3 /home/leizy/24sum/CFRL/src/main_model/run/train.py \
    --model_name_or_path /home/leizy/24fall/src/main_model/T5-base \
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file /home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt_510/1_step/jsonl_data/train.json \
    --validation_file /home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt_510/1_step/jsonl_data/val.json \
    --output_dir /home/leizy/24sum/CFRL/src/main_model/trained_model/with_prompt_510/T5_1_step \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate