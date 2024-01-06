#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/llama2-70b'
TASKS='kowikitable'
GPU_NO=2

CURRENT_TRAINED_TOKENS=1t
MODEL="meta-llama/Llama-2-70b-hf"

NUM_SHOTS=0

CURRENT_TRAINED_TOKENS="$MODEL-test-$CURRENT_TRAINED_TOKENS"
echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL,load_in_8bit=True \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 1 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

