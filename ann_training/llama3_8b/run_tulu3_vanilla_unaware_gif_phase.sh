#!/usr/bin/env bash
set -euo pipefail

# ./ann_training/llama3_8b/run_tulu3_vanilla_unaware_gif_phase.sh

PROJECT_ROOT="/home/wangwenkang/SNN"
GPU_DEVICES="0,1,2"

cd "$PROJECT_ROOT"

run_step() {
  local label="$1"
  local script_path="$2"

  if [[ ! -x "$script_path" ]]; then
    echo "Missing or non-executable script: $script_path" >&2
    return 127
  fi

  echo "============================================================"
  echo "[START] $label"
  echo "CUDA_VISIBLE_DEVICES=$GPU_DEVICES $script_path"
  echo "============================================================"

  local status
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$script_path"
  status=$?
  set -e

  if (( status != 0 )); then
    echo "[FAILED] $label (exit_code=$status)" >&2
    return "$status"
  fi

  echo "[DONE] $label"
}

declare -a FAILED_STEPS=()

# if ! run_step \
#   "Tulu 3 vanilla + unaware" \
#   "./ann_training/llama3_8b/train_discover_prefix_and_evaluate_llama3_8b_tulu3_vanilla_unaware_LEARNING_RATE.sh"; then
#   FAILED_STEPS+=("Tulu 3 vanilla + unaware")
# fi

# echo "[WAIT] Waiting 10 seconds before GIF-aware training."
# sleep 10

if ! run_step \
  "Tulu 3 GIF-aware" \
  "./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_gif_aware.sh"; then
  FAILED_STEPS+=("Tulu 3 GIF-aware")
fi

echo "[WAIT] Waiting 10 seconds before Phase-aware training."
sleep 10

if ! run_step \
  "Tulu 3 Phase-aware" \
  "./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_phase_aware.sh"; then
  FAILED_STEPS+=("Tulu 3 Phase-aware")
fi

if (( ${#FAILED_STEPS[@]} > 0 )); then
  printf 'Completed with failed steps:\n'
  printf '  %s\n' "${FAILED_STEPS[@]}"
  exit 1
fi

echo "All Tulu 3 ANN training and lm-eval jobs completed successfully."
