#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml"

# HTGE temperature values to run sequentially.
HTGE_TS=(
  6.0
  8.0
  10.0
  12.0
  14.0
  18.0
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_gif_aware_htge_t.sh
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

normalize_htge_t() {
  python3 -c '
from decimal import Decimal, InvalidOperation
import sys
try:
    value = Decimal(sys.argv[1])
except InvalidOperation as exc:
    raise SystemExit(f"Invalid HTGE_T: {sys.argv[1]}") from exc
if not value.is_finite() or value <= 0:
    raise SystemExit("HTGE_T must be a finite number greater than 0")
print(format(value.normalize(), "f"))
' "$1"
}

set_htge_t() {
  local run_cfg="$1"
  local htge_t="$2"
  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
text = path.read_text(encoding="utf-8")
updated, count = re.subn(r"(?m)^(  htge_t:)\s*\S+\s*$", rf"\1 {sys.argv[2]}", text)
if count != 1:
    raise SystemExit(f"Expected exactly one gif.htge_t entry, found {count}")
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
values = re.findall(r"(?m)^  htge_t:\s*(\S+)\s*$", updated)
if len(values) != 1 or Decimal(values[0]) != expected:
    raise SystemExit(f"gif.htge_t mismatch: expected={expected}, actual={values}")
print(f"Confirmed gif.htge_t={values[0]} in {path}")
' "$run_cfg" "$htge_t"
}

run_one_htge_t() {
  local htge_t="$1" normalized_htge_t lock_key lock_file run_cfg train_status eval_status
  normalized_htge_t="$(normalize_htge_t "$htge_t")"
  lock_key="${normalized_htge_t//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-llama3-8b-tulu3-gif-aware-htge-t-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using htge_t=$htge_t" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-llama3-8b-tulu3-gif-aware-htge-t-${lock_key}.XXXXXX.yaml")"
  cleanup_run_cfg() { rm -f -- "$run_cfg"; }

  if ! cp -- "$SOURCE_CFG" "$run_cfg" || ! set_htge_t "$run_cfg" "$htge_t"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[TRAIN] htge_t=$htge_t GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  torchrun --standalone --nproc_per_node="$NGPU" scripts/train_ann.py --config "$run_cfg"
  train_status=$?
  set -e
  if (( train_status != 0 )); then
    echo "[FAILED] ANN training failed; skipping lm-eval for htge_t=$htge_t" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  # Re-apply and verify the per-instance override before evaluation.
  if ! set_htge_t "$run_cfg" "$htge_t"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[EVAL] htge_t=$htge_t GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
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

declare -a FAILED_HTGE_TS=()
declare -a SUCCESSFUL_HTGE_TS=()
echo "Llama-3-8B Tulu-3 GIF-Aware ANN Training + lm-eval"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "HTGE T values        : ${HTGE_TS[*]}"

for htge_t in "${HTGE_TS[@]}"; do
  if run_one_htge_t "$htge_t"; then
    SUCCESSFUL_HTGE_TS+=("$htge_t")
  else
    FAILED_HTGE_TS+=("$htge_t")
    echo "[WARNING] htge_t=$htge_t failed; continuing." >&2
  fi
done

echo "Completed ${#HTGE_TS[@]} HTGE-T runs: ${#SUCCESSFUL_HTGE_TS[@]} succeeded, ${#FAILED_HTGE_TS[@]} failed."
if (( ${#FAILED_HTGE_TS[@]} > 0 )); then
  printf 'Failed HTGE T values:\n  %s\n' "${FAILED_HTGE_TS[@]}"
  exit 1
fi
