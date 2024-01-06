#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/llama2koenco-7b-test'
TASKS='hellaswag,arc_challenge,squad2'
GPU_NO=4

CURRENT_TRAINED_TOKENS=50v
# MODEL='meta-llama/Llama-2-13b-hf'
MODEL="Taekyoon/llama2-koenco-7b-test"

NUM_SHOTS=5

CURRENT_TRAINED_TOKENS="$MODEL-comment-only-$CURRENT_TRAINED_TOKENS"
echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS


python main.py \
--model gpt2 \
--model_args pretrained=$MODEL,revision=stage0_20b_150k \
--tasks $TASKS \
--num_fewshot $NUM_SHOTS \
--device cuda:$GPU_NO \
--batch_size 8 \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

