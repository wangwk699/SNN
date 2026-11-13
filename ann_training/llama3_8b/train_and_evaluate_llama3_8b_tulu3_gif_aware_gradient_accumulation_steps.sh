#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml"

# Gradient accumulation steps to run sequentially.
STEPS=(
  8
  32
  48
  64
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2 ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_gif_aware_gradient_accumulation_steps.sh
gpu_devices="${CUDA_VISIBLE_DEVICES:-2,3}"

cd "$PROJECT_ROOT"

if [[ ! -f "$SOURCE_CFG" ]]; then
  echo "Missing generated config: $PROJECT_ROOT/$SOURCE_CFG" >&2
  exit 1
fi
if [[ -z "$gpu_devices" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be non-empty" >&2
  exit 1
fi

IFS=',' read -r -a gpu_list <<< "$gpu_devices"
NGPU="${#gpu_list[@]}"
if (( NGPU <= 0 )); then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
  exit 1
fi
for gpu in "${gpu_list[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id in CUDA_VISIBLE_DEVICES: $gpu" >&2
    exit 1
  fi
done
export CUDA_VISIBLE_DEVICES="$gpu_devices"

normalize_gradient_accumulation_steps() {
  python3 -c '
import sys
try:
    value = int(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"Invalid GRADIENT_ACCUMULATION_STEPS: {sys.argv[1]}") from exc
if value <= 0:
    raise SystemExit("GRADIENT_ACCUMULATION_STEPS must be a positive integer")
print(value)
' "$1"
}

set_gradient_accumulation_steps() {
  local run_cfg="$1"
  local gradient_accumulation_steps="$2"
  python3 -c '
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
text = path.read_text(encoding="utf-8")
updated, count = re.subn(r"(?m)^(  gradient_accumulation_steps:)\s*\S+\s*$", rf"\1 {sys.argv[2]}", text)
if count != 1:
    raise SystemExit(f"Expected exactly one training.gradient_accumulation_steps entry, found {count}")
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
values = re.findall(r"(?m)^  gradient_accumulation_steps:\s*(\S+)\s*$", updated)
if len(values) != 1 or not values[0].isdigit() or int(values[0]) != expected:
    raise SystemExit(f"training.gradient_accumulation_steps mismatch: expected={expected}, actual={values}")
print(f"Confirmed training.gradient_accumulation_steps={values[0]} in {path}")
' "$run_cfg" "$gradient_accumulation_steps"
}

run_one_gradient_accumulation_steps() {
  local gradient_accumulation_steps="$1" normalized_gradient_accumulation_steps lock_key lock_file run_cfg train_status eval_status
  normalized_gradient_accumulation_steps="$(normalize_gradient_accumulation_steps "$gradient_accumulation_steps")"
  lock_key="${normalized_gradient_accumulation_steps//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-llama3-8b-tulu3-gif-aware-gradient-accumulation-steps-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using gradient_accumulation_steps=$gradient_accumulation_steps" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-llama3-8b-tulu3-gif-aware-gradient-accumulation-steps-${lock_key}.XXXXXX.yaml")"
  cleanup_run_cfg() { rm -f -- "$run_cfg"; }

  if ! cp -- "$SOURCE_CFG" "$run_cfg" || ! set_gradient_accumulation_steps "$run_cfg" "$gradient_accumulation_steps"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[TRAIN] gradient_accumulation_steps=$gradient_accumulation_steps GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  torchrun --standalone --nproc_per_node="$NGPU" scripts/train_ann.py --config "$run_cfg"
  train_status=$?
  set -e
  if (( train_status != 0 )); then
    echo "[FAILED] ANN training failed; skipping lm-eval for gradient_accumulation_steps=$gradient_accumulation_steps" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  # Re-apply and verify the per-instance override before evaluation.
  if ! set_gradient_accumulation_steps "$run_cfg" "$gradient_accumulation_steps"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[EVAL] gradient_accumulation_steps=$gradient_accumulation_steps GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  accelerate launch --num_processes "$NGPU" scripts/evaluate_lm_harness.py \
    --config "$run_cfg" --neuron ann
  eval_status=$?
  set -e

  cleanup_run_cfg
  flock -u 9
  exec 9>&-
  return "$eval_status"
}

declare -a FAILED_STEPS=()
declare -a SUCCESSFUL_STEPS=()
echo "Llama-3-8B Tulu-3 GIF-Aware ANN Training + lm-eval"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Gradient accumulation steps        : ${STEPS[*]}"

for gradient_accumulation_steps in "${STEPS[@]}"; do
  if run_one_gradient_accumulation_steps "$gradient_accumulation_steps"; then
    SUCCESSFUL_STEPS+=("$gradient_accumulation_steps")
  else
    FAILED_STEPS+=("$gradient_accumulation_steps")
    echo "[WARNING] gradient_accumulation_steps=$gradient_accumulation_steps failed; continuing." >&2
  fi
done

echo "Completed ${#STEPS[@]} gradient-accumulation-steps runs: ${#SUCCESSFUL_STEPS[@]} succeeded, ${#FAILED_STEPS[@]} failed."
if (( ${#FAILED_STEPS[@]} > 0 )); then
  printf 'Failed gradient accumulation steps:\n  %s\n' "${FAILED_STEPS[@]}"
  exit 1
fi
