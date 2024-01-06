export TOKENIZERS_PARALLELISM=false

# export CUDA_VISIBLE_DEVICES="5,6,7"

export HF_DATASETS_CACHE="/mnt/ddn/cache/hf_dataset_cache"
export TRANSFORMERS_CACHE="/mnt/fr20tb/tgchoi/cache/hf_model_cache"

RESULT_DIR='results/llama-2-ko-exp'
MODEL=$1
TASKS='kobest_hellaswag,kobest_wic,kobest_copa,kobest_boolq,kobest_sentineg,nsmc,korquad,klue_mrc,klue_sts,klue_nli,klue_ynat,kohatespeech_apeach,pawsx_ko'
# TASKS='kobest_hellaswag,kobest_wic,kobest_copa,kobest_boolq,kobest_sentineg,nsmc,korquad,kohatespeech_apeach,pawsx_ko'
# TASKS='kobest_hellaswag,kobest_copa'
GPU_NO=$2

CURRENT_TRAINED_TOKENS=$4
echo "mkdir -p $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS

# MODEL="../news_gpt_research/KoAlpaca/llama2-koenalpaca_7b_1t"
# MODEL="../news_gpt_research/peft/examples/int8_training/llama2_koen_7b_lora_midhighlow4_w_enko_trans_3eps"
# MODEL='hyunseoki/ko-en-llama2-13b'
# MODEL='Taekyoon/codellama-koenco-7b-test'
# MODEL='meta-llama/Llama-2-7b-hf'
# MODEL="../news_gpt_research/KoAlpaca/llama2-org-alpaca_7b_test"
# MODEL='EleutherAI/polyglot-ko-5.8b'
# MODEL='../news_gpt_research/EasyLM/models/llama2-koentrans-7b-test'

echo $MODEL

# python main.py \
# --model gpt2 \
# --model_args pretrained=$MODEL,revision=blog_20b_300k \
# --tasks $TASKS \
# --num_fewshot 0 \
# --device cuda:$GPU_NO \
# --output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/00_shot.json

# ,revision=stage0_20b_150k

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL,revision=$3 \
--tasks $TASKS \
--num_fewshot 5 \
--device cuda:$GPU_NO \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/05_shot.json

# python main.py \
# --model gpt2 \
# --model_args pretrained=$MODEL,revision=blog_20b_300k \
# --tasks $TASKS \
# --num_fewshot 10 \
# --device cuda:$GPU_NO \
# --output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/10_shot.json

# python main.py \
# --model gpt2 \
# --model_args pretrained=$MODEL \
# --tasks $TASKS \
# --num_fewshot 5 \
# --device auto \
# --output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/5_shot.json
