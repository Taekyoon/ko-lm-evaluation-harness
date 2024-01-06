#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/llama2-13b'
TASKS='korquad'
GPU_NO=4

CURRENT_TRAINED_TOKENS=1t
MODEL="meta-llama/Llama-2-13b-hf"

NUM_SHOTS=0

CURRENT_TRAINED_TOKENS="$MODEL-test-$CURRENT_TRAINED_TOKENS"
echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 8 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

