#!/bin/bash
export MSYS_NO_PATHCONV=1

uv run modal run --detach scripts/modal_train.py::train_remote -- \
--algo ipo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_ipo_v1 \
--beta 0.1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 100 \
--eval_interval 100 \
--save_interval 100 \
--no-wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_ipo_v1