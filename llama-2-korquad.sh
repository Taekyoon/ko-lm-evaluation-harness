#!/bin/bash

export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_CACHE="/mnt/ddn/cache/hf_dataset_cache"
export TRANSFORMERS_CACHE="/mnt/fr20tb/tgchoi/cache/hf_model_cache"

RESULT_DIR='results/llama2'
TASKS='nsmc'
GPU_NO=1

CURRENT_TRAINED_TOKENS=1t
MODEL="Taekyoon/Solar-ko-10b-test"

NUM_SHOTS=5

CURRENT_TRAINED_TOKENS="$MODEL-test-$CURRENT_TRAINED_TOKENS"
echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL,revision=koen_stage0_20b_100k \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 5 \
--no_cache \
--limit 100 \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

