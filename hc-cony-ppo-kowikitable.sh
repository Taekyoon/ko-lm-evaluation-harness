#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2


RESULT_DIR='results/hc-moon-ppo'
TASKS='kowikitable'
GPU_NO=0

CURRENT_TRAINED_TOKENS=180b
MODEL="../base_models/hc_cony_ppo_mod"

NUM_SHOTS=3

CURRENT_TRAINED_TOKENS="$MODEL-$CURRENT_TRAINED_TOKENS"
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
--limit 100 \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

