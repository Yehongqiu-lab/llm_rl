#!/bin/bash
export MSYS_NO_PATHCONV=1

model_version="v6"
modelchar="dpo"
betachar="0005"
step="000297"


uv run modal run --detach scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_${modelchar}_beta${betachar}_${model_version}/checkpoints/step_${step}/adapter \
  --prompts_jsonl /vol/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/${modelchar}.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0