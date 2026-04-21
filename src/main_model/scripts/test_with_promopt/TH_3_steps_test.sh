#!/bin/bash


# TH 1 step
# Base directory containing the 16 directories
BASE_DIR="/home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt/3_steps/TH_S"
MODELS_DIR="/home/leizy/24sum/CFRL/src/main_model/trained_model/with_prompt/trained_together/T5_3_steps"
OUTPUT_BASE_DIR="/home/leizy/24sum/CFRL/data/train_together_results/predicted_results/with_prompt/TH/3_steps"

for DIR in "$BASE_DIR"/*/; do
    # Extract the directory name (without the full path)
    DIR_NAME=$(basename "$DIR")

    OUTPUT_DIR="$OUTPUT_BASE_DIR/$DIR_NAME"

    python3 /home/leizy/24sum/CFRL/src/main_model/run/test.py \
        -t "$DIR" \
        -m "$MODELS_DIR" \
        -s "$OUTPUT_DIR" \
        -d "$DIR_NAME" \
        --model_name T5 \
        -b 16
done