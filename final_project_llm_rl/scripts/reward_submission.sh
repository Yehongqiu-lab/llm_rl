#!/bin/bash
export MSYS_NO_PATHCONV=1

reward_model_run="wildchat_min4_judged_5k_reward_model_v1"
step="step_000445"
model_name="dpo"

uv run modal run --detach scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/${reward_model_run}/checkpoints/${step}/adapter \
  --prefs_jsonl /vol/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl /vol/submissions/public_test_pref_scores.jsonl \
  --max_prompt_tokens 700 \
  --max_response_tokens 512