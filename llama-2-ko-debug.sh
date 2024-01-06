#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/debug_test'
# ,kobest_copa,arc_challenge,squad2,korquad,kobest_hellaswag'
TASKS='korquad' #kobest_wic' #hellaswag'
GPU_NO=0

CURRENT_TRAINED_TOKENS=1t
# MODEL="EleutherAI/polyglot-ko-12.8b"
# MODEL="Taekyoon/llama2-koen-7b-test"
MODEL="../news_gpt_research/peft/examples/int8_training/llama2_koen_lora_highlow4_w_enko_trans_1eps"
# MODEL="Taekyoon/llama2-koen-7b-test"

NUM_SHOTS=5

CURRENT_TRAINED_TOKENS="$MODEL-comment-only-$CURRENT_TRAINED_TOKENS"
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
--limit 1000 \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${NUM_SHOTS}_shot.json

