#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/kogpt'
TASKS='korquad'
GPU_NO=3

MODEL="kakaobrain/kogpt"

NUM_SHOTS=0

CURRENT_TRAINED_TOKENS="$MODEL-test"
echo "mkdir -p $RESULT_DIR"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL,revision=KoGPT6B-ryan1.5b \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 8 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

