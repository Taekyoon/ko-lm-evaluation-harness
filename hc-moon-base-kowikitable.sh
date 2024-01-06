#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/hc-moon-base-korquad'
TASKS='korquad'
GPU_NO=0

CURRENT_TRAINED_TOKENS=180b
MODEL="../base_models/hc_moon_base"

NUM_SHOTS=5

CURRENT_TRAINED_TOKENS="$MODEL-$CURRENT_TRAINED_TOKENS"
echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 1 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

