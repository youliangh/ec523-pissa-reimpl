#!/bin/bash
export PYTHONPATH=$(pwd)

dataset="math10k"  # choice ["stack_exchange_paired", "xSum", "math10k", "MetaMath", "commonsense170k", "commonsense15k"]
output_dir="output/$dataset"
model_path=XXX  # path to the base model

dir_count=1
printf -v dname "%05d" "$dir_count"
while [ -d "$output_dir/$dname" ]
do
  ((dir_count++))
  printf -v dname "%05d" "$dir_count"
done

printf "Creating output directory %s...\n" "$output_dir/$dname"
mkdir -p "$output_dir/$dname"
output_dir=$output_dir/$dname

# Not tested
CUDA_VISIBLE_DEVICES="0" python finetune.py \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --dataset $dataset \
    --logging_steps 1 \
    --save_strategy no \
    --data_seed 42 \
    --save_steps 1562 \
    --save_total_limit 40 \
    --evaluation_strategy no \
    --eval_dataset_size 1 \
    --max_eval_samples 1 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 512 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval False \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --full_finetune False \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --source_max_len 128 \
    --target_max_len 384 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --eval_steps 1573 \
    --learning_rate 0.0003 \
    --max_steps -1 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0 \
    --weight_decay 0.0 \
    --include_num_input_tokens_seen \
    --seed 0 \
    --pretokenize False \
    --eval_test True