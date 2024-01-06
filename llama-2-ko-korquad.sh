#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/llama2ko_korquad_single_test'
TASKS='korquad'
GPU_NO=4

CURRENT_TRAINED_TOKENS=65
MODEL="beomi/llama-2-ko-7b"

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
--batch_size 8 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${TASKS}_${NUM_SHOTS}_shot.json

