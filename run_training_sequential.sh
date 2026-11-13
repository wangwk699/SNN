#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

tasks=(
  ./ann_training/qwen3_8b/train_and_evaluate_qwen3_8b_gif_aware.sh
  ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_phase_aware.sh
  ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_gif_aware.sh
)
failed=0

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  echo "[START] $task"
  if CUDA_VISIBLE_DEVICES=0,3,4,7 "$task"; then
    echo "[SUCCESS] $task"
  else
    status=$?
    echo "[WARNING] $task failed (exit_code=$status); continuing." >&2
    failed=$((failed + 1))
  fi

  if (( index + 1 < ${#tasks[@]} )); then
    sleep 5
  fi
done

echo "[SUMMARY] Total: ${#tasks[@]}, Failed: $failed"
if (( failed > 0 )); then
  exit 1
fi
