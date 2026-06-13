#!/bin/bash
export MSYS_NO_PATHCONV=1

uv run python student_autograder/run_local_autograder.py \
  --submission_dir llm_rl_final_proj_public_submission \
  --output_json student_autograder_results.json