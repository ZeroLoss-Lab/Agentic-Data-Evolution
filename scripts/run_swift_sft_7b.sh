#!/usr/bin/env bash
set -euo pipefail

# Distributed training: 8 GPUs, Qwen2.5-7B
# Required environment variables:
#   MODEL      - Path to base model
#   DATASET    - Path to training dataset
#   OUTPUT_DIR - Directory for training outputs

PYTORCH_ALLOC_CONF="expandable_segments:True" \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model ${MODEL} \
    --tuner_type full \
    --dataset "${DATASET}" \
    --system "You are a helpful assistant." \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 16 \
    --split_dataset_ratio 0.05 \
    --eval_steps 25 \
    --save_steps 50 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 1 \
    --save_only_model true \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
