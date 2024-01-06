#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/kogpt'
TASKS='kohatespeech_apeach'
GPU_NO=6

MODEL="Taekyoon/llama2-koen-7b-test"

NUM_SHOTS=5

CURRENT_TRAINED_TOKENS="$MODEL-test"
echo "mkdir -p $RESULT_DIR"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL,revision=stage1_20b_250k \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 8 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

