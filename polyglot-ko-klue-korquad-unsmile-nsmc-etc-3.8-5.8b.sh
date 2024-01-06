#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/korean_origin_bench'
TASKS='korquad'
GPU_NO=7

MODELS=("5.8b")

for CURRENT_MODEL in "${MODELS[@]}"
do
  CURRENT_TRAINED_TOKENS="polyglot-ko-korquad-test-$CURRENT_MODEL"
  echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
  mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

  MODEL="EleutherAI/polyglot-ko-$CURRENT_MODEL"

  for num_fewshot in 5
  do
    python main.py \
    --model gpt2 \
    --model_args pretrained=$MODEL \
    --tasks $TASKS \
    --num_fewshot $num_fewshot \
    --device cuda:$GPU_NO \
    --batch_size 32 \
    --no_cache \
    --output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${num_fewshot}_shot.json
  done
done

#     --limit 100 \
