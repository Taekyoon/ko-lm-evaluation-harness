#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/polyglot-12.8'
TASKS='kowikitable'
GPU_NO=2

CURRENT_TRAINED_TOKENS=1t
MODEL="EleutherAI/polyglot-ko-12.8b"

NUM_SHOTS=5

CURRENT_TRAINED_TOKENS="$MODEL-test-$CURRENT_TRAINED_TOKENS"
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

