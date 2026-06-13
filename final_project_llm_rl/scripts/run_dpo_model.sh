#!/bin/bash
export MSYS_NO_PATHCONV=1

model_version="v2"
train_epochs=2
lr_value=5e-5
beta_value=0.005
betachar="005"

uv run modal run --detach scripts/modal_train.py::train_remote -- \
--algo dpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_dpo_beta${betachar}_${model_version} \
--beta ${beta_value} \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 4 \
--lr ${lr_value} \
--num_train_epochs ${train_epochs} \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 100 \
--eval_interval 100 \
--save_interval 100 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_dpo_beta${betachar}_${model_version}