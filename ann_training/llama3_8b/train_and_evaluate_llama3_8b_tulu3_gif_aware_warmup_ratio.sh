#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml"

# Warmup ratios to run sequentially.
WARMUP_RATIOS=(
  0.03
  0.05
  0.1
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_gif_aware_warmup_ratio.sh
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

normalize_warmup_ratio() {
  python3 -c '
from decimal import Decimal, InvalidOperation
import sys
try:
    value = Decimal(sys.argv[1])
except InvalidOperation as exc:
    raise SystemExit(f"Invalid WARMUP_RATIO: {sys.argv[1]}") from exc
if not value.is_finite() or value < 0 or value > 1:
    raise SystemExit("WARMUP_RATIO must be a finite number in [0, 1]")
print(format(value.normalize(), "f"))
' "$1"
}

set_warmup_ratio() {
  local run_cfg="$1"
  local warmup_ratio="$2"
  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
text = path.read_text(encoding="utf-8")
updated, count = re.subn(r"(?m)^(  warmup_ratio:)\s*\S+\s*$", rf"\1 {sys.argv[2]}", text)
if count != 1:
    raise SystemExit(f"Expected exactly one training.warmup_ratio entry, found {count}")
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
values = re.findall(r"(?m)^  warmup_ratio:\s*(\S+)\s*$", updated)
if len(values) != 1 or Decimal(values[0]) != expected:
    raise SystemExit(f"training.warmup_ratio mismatch: expected={expected}, actual={values}")
print(f"Confirmed training.warmup_ratio={values[0]} in {path}")
' "$run_cfg" "$warmup_ratio"
}

run_one_warmup_ratio() {
  local warmup_ratio="$1" normalized_warmup_ratio lock_key lock_file run_cfg train_status eval_status
  normalized_warmup_ratio="$(normalize_warmup_ratio "$warmup_ratio")"
  lock_key="${normalized_warmup_ratio//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-llama3-8b-tulu3-gif-aware-warmup-ratio-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using warmup_ratio=$warmup_ratio" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-llama3-8b-tulu3-gif-aware-warmup-ratio-${lock_key}.XXXXXX.yaml")"
  cleanup_run_cfg() { rm -f -- "$run_cfg"; }

  if ! cp -- "$SOURCE_CFG" "$run_cfg" || ! set_warmup_ratio "$run_cfg" "$warmup_ratio"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[TRAIN] warmup_ratio=$warmup_ratio GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  torchrun --standalone --nproc_per_node="$NGPU" scripts/train_ann.py --config "$run_cfg"
  train_status=$?
  set -e
  if (( train_status != 0 )); then
    echo "[FAILED] ANN training failed; skipping lm-eval for warmup_ratio=$warmup_ratio" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  # Re-apply and verify the per-instance override before evaluation.
  if ! set_warmup_ratio "$run_cfg" "$warmup_ratio"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[EVAL] warmup_ratio=$warmup_ratio GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
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

declare -a FAILED_WARMUP_RATIOS=()
declare -a SUCCESSFUL_WARMUP_RATIOS=()
echo "Llama-3-8B Tulu-3 GIF-Aware ANN Training + lm-eval"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Warmup ratios        : ${WARMUP_RATIOS[*]}"

for warmup_ratio in "${WARMUP_RATIOS[@]}"; do
  if run_one_warmup_ratio "$warmup_ratio"; then
    SUCCESSFUL_WARMUP_RATIOS+=("$warmup_ratio")
  else
    FAILED_WARMUP_RATIOS+=("$warmup_ratio")
    echo "[WARNING] warmup_ratio=$warmup_ratio failed; continuing." >&2
  fi
done

echo "Completed ${#WARMUP_RATIOS[@]} warmup-ratio runs: ${#SUCCESSFUL_WARMUP_RATIOS[@]} succeeded, ${#FAILED_WARMUP_RATIOS[@]} failed."
if (( ${#FAILED_WARMUP_RATIOS[@]} > 0 )); then
  printf 'Failed warmup ratios:\n  %s\n' "${FAILED_WARMUP_RATIOS[@]}"
  exit 1
fi
