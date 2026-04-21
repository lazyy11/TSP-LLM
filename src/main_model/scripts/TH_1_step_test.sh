#!/bin/bash


# TH 1 step
# Base directory containing the 16 directories
BASE_DIR="/home/leizy/24sum/CFRL/data/no_prompt/1_step/TH_S"

MODELS_DIR="/home/leizy/24sum/CFRL/src/main_model/trained_model/no_prompt/T5_1_step"
OUTPUT_BASE_DIR="/home/leizy/24sum/CFRL/data/predictied_results/no_prompt/TH/1_step"

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