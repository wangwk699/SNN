#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml"

# Learning rates to run sequentially.
LEARNING_RATES=(
    2.0e-06
    3.0e-06
    4.0e-06
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,2,3,4 ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_gif_aware.sh
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

normalize_learning_rate() {
  python3 -c '
from decimal import Decimal, InvalidOperation
import sys
try:
    value = Decimal(sys.argv[1])
except InvalidOperation as exc:
    raise SystemExit(f"Invalid LEARNING_RATE: {sys.argv[1]}") from exc
if not value.is_finite() or value <= 0:
    raise SystemExit("LEARNING_RATE must be a positive finite number")
print(format(value.normalize(), "E"))
' "$1"
}

set_learning_rate() {
  local run_cfg="$1"
  local learning_rate="$2"
  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
text = path.read_text(encoding="utf-8")
updated, count = re.subn(r"(?m)^(  learning_rate:)\s*\S+\s*$", rf"\1 {sys.argv[2]}", text)
if count != 1:
    raise SystemExit(f"Expected exactly one training.learning_rate entry, found {count}")
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
values = re.findall(r"(?m)^  learning_rate:\s*(\S+)\s*$", updated)
if len(values) != 1 or Decimal(values[0]) != expected:
    raise SystemExit(f"training.learning_rate mismatch: expected={expected}, actual={values}")
print(f"Confirmed training.learning_rate={values[0]} in {path}")
' "$run_cfg" "$learning_rate"
}

run_one_learning_rate() {
  local learning_rate="$1" normalized_learning_rate lock_key lock_file run_cfg train_status eval_status
  normalized_learning_rate="$(normalize_learning_rate "$learning_rate")"
  lock_key="${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-llama3-8b-tulu3-gif-aware-lr-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using learning_rate=$learning_rate" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-llama3-8b-tulu3-gif-aware-${lock_key}.XXXXXX.yaml")"
  cleanup_run_cfg() { rm -f -- "$run_cfg"; }

  if ! cp -- "$SOURCE_CFG" "$run_cfg" || ! set_learning_rate "$run_cfg" "$learning_rate"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[TRAIN] learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  torchrun --standalone --nproc_per_node="$NGPU" scripts/train_ann.py --config "$run_cfg"
  train_status=$?
  set -e
  if (( train_status != 0 )); then
    echo "[FAILED] ANN training failed; skipping lm-eval for learning_rate=$learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  # Re-apply and verify the per-instance override before evaluation.
  if ! set_learning_rate "$run_cfg" "$learning_rate"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[EVAL] learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
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

declare -a FAILED_LEARNING_RATES=()
declare -a SUCCESSFUL_LEARNING_RATES=()
echo "Llama-3-8B Tulu-3 GIF-Aware ANN Training + lm-eval"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Learning rates       : ${LEARNING_RATES[*]}"

for learning_rate in "${LEARNING_RATES[@]}"; do
  if run_one_learning_rate "$learning_rate"; then
    SUCCESSFUL_LEARNING_RATES+=("$learning_rate")
  else
    FAILED_LEARNING_RATES+=("$learning_rate")
    echo "[WARNING] learning_rate=$learning_rate failed; continuing." >&2
  fi
done

echo "Completed ${#LEARNING_RATES[@]} learning-rate runs: ${#SUCCESSFUL_LEARNING_RATES[@]} succeeded, ${#FAILED_LEARNING_RATES[@]} failed."
if (( ${#FAILED_LEARNING_RATES[@]} > 0 )); then
  printf 'Failed learning rates:\n  %s\n' "${FAILED_LEARNING_RATES[@]}"
  exit 1
fi
