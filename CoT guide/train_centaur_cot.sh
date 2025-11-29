#!/bin/bash

# Train Centaur-CoT on the CoT-augmented Psych-101 dataset
# Requires: psych101_cot_train.jsonl (generated from create_cot_dataset_*.py)
# Hardware: A100 80GB GPU recommended
# Time: 48-96 hours

python finetune_cot.py \
  --seed 100 \
  --model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
  --cot_data_path "psych101_cot_train.jsonl" \
  --max_seq_len 32768 \
  --num_train_epochs 5 \
  --log_level "info" \
  --logging_strategy "steps" \
  --logging_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 999999 \
  --save_strategy "steps" \
  --save_steps 100 \
  --learning_rate 5e-5 \
  --optim "adamw_8bit" \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --output_dir "centaur-cot-70b" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32

