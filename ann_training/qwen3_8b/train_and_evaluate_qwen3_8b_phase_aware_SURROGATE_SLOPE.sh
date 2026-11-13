#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml"

# Surrogate slopes to run sequentially.
SLOPES=(
  3.5
  4.5
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=2,3,4,5 ./ann_training/qwen3_8b/train_and_evaluate_qwen3_8b_phase_aware_SURROGATE_SLOPE.sh
gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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

if ((NGPU <= 0)); then
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

echo "============================================================"
echo "Qwen3-8B Phase-Aware ANN Training + Evaluation"
echo "============================================================"
echo "Project root         : $PROJECT_ROOT"
echo "Source config        : $SOURCE_CFG"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Surrogate slopes     : ${SLOPES[*]}"
echo "Total slope runs     : ${#SLOPES[@]}"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

normalize_slope() {
  python3 -c '
from decimal import Decimal, InvalidOperation
import sys

try:
    value = Decimal(sys.argv[1])
except InvalidOperation as exc:
    raise SystemExit(f"Invalid surrogate slope: {sys.argv[1]}") from exc

if not value.is_finite() or value <= 0:
    raise SystemExit("surrogate_slope must be a positive finite number")

print(format(value.normalize(), "E"))
' "$1"
}

override_slope() {
  local run_cfg="$1"
  local slope="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
slope = sys.argv[2]
text = path.read_text(encoding="utf-8")

updated, count = re.subn(
    r"(?m)^(  surrogate_slope:)\s*\S+\s*$",
    rf"\g<1> {slope}",
    text,
)

if count != 1:
    raise SystemExit(
        f"Expected exactly one phase.surrogate_slope entry, found {count}"
    )

path.write_text(
    updated + ("" if updated.endswith("\n") else "\n"),
    encoding="utf-8",
)
' "$run_cfg" "$slope"
}

assert_slope() {
  local run_cfg="$1"
  local slope="$2"

  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
text = path.read_text(encoding="utf-8")
values = re.findall(r"(?m)^  surrogate_slope:\s*(\S+)\s*$", text)

if len(values) != 1:
    raise SystemExit(
        f"Expected exactly one phase.surrogate_slope entry, found {len(values)}"
    )

actual = Decimal(values[0])
if actual != expected:
    raise SystemExit(
        f"phase.surrogate_slope mismatch: expected={expected}, actual={actual}"
    )

print(f"Confirmed phase.surrogate_slope={values[0]} in {path}")
' "$run_cfg" "$slope"
}

# ---------------------------------------------------------------------------
# Run one surrogate-slope experiment
# ---------------------------------------------------------------------------

run_one_slope() {
  local slope="$1"
  local normalized_slope
  local lock_key
  local lock_file
  local run_cfg
  local train_status
  local eval_status

  echo
  echo "============================================================"
  echo "Starting surrogate_slope=$slope"
  echo "GPUs=$CUDA_VISIBLE_DEVICES"
  echo "NGPU=$NGPU"
  echo "============================================================"

  if ! normalized_slope="$(normalize_slope "$slope")"; then
    echo "[FAILED] Invalid surrogate slope: $slope" >&2
    return 1
  fi

  lock_key="${normalized_slope//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-qwen3-8b-phase-aware-slope-${lock_key}.lock"

  # Prevent concurrent writers for the same slope.
  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using surrogate_slope=$slope" >&2
    exec 9>&-
    return 1
  fi

  # Create an independent config for this slope.
  run_cfg="$(mktemp "/tmp/snn-qwen3-8b-phase-aware-slope-${lock_key}.XXXXXX.yaml")"
  echo "Per-instance config: $run_cfg"

  cleanup_run_cfg() {
    rm -f -- "$run_cfg"
  }

  if ! cp -- "$SOURCE_CFG" "$run_cfg"; then
    echo "[FAILED] Failed to create per-instance config" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  if ! override_slope "$run_cfg" "$slope" || ! assert_slope "$run_cfg" "$slope"; then
    echo "[FAILED] surrogate_slope configuration failed for $slope" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo
  echo "[TRAIN] surrogate_slope=$slope"
  echo "[TRAIN] torchrun --nproc_per_node=$NGPU"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  set +e
  torchrun \
    --standalone \
    --nproc_per_node="$NGPU" \
    scripts/train_ann.py \
    --config "$run_cfg"
  train_status=$?
  set -e

  if ((train_status != 0)); then
    echo "[FAILED] ANN training failed: surrogate_slope=$slope, exit_code=$train_status" >&2
    echo "Skipping evaluation and continuing to next surrogate slope." >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  echo "[TRAIN] Completed successfully for surrogate_slope=$slope"

  # Re-apply and verify in case training modified the temporary configuration.
  if ! override_slope "$run_cfg" "$slope" || ! assert_slope "$run_cfg" "$slope"; then
    echo "[FAILED] surrogate_slope verification failed before evaluation" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo
  echo "[EVAL] surrogate_slope=$slope"
  echo "[EVAL] accelerate --num_processes $NGPU"

  set +e
  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$run_cfg" \
    --neuron ann
  eval_status=$?
  set -e

  if ((eval_status != 0)); then
    echo "[FAILED] ANN evaluation failed: surrogate_slope=$slope, exit_code=$eval_status" >&2
    echo "Continuing to next surrogate slope." >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$eval_status"
  fi

  echo "[SUCCESS] surrogate_slope=$slope"
  cleanup_run_cfg
  flock -u 9
  exec 9>&-
  return 0
}

# ---------------------------------------------------------------------------
# Run all surrogate slopes sequentially
# ---------------------------------------------------------------------------

declare -a FAILED_SLOPES=()
declare -a SUCCESSFUL_SLOPES=()
TOTAL_RUNS="${#SLOPES[@]}"
CURRENT_RUN=0

for slope in "${SLOPES[@]}"; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  echo
  echo "################################################################"
  echo "# Experiment $CURRENT_RUN / $TOTAL_RUNS"
  echo "# surrogate_slope=$slope"
  echo "################################################################"

  if run_one_slope "$slope"; then
    SUCCESSFUL_SLOPES+=("$slope")
  else
    FAILED_SLOPES+=("$slope")
    echo "[WARNING] surrogate_slope=$slope failed; continuing to the next slope." >&2
  fi
done

echo
echo "================================================================"
echo "All surrogate-slope experiments have finished."
echo "================================================================"
echo "Successful slopes: ${SUCCESSFUL_SLOPES[*]:-None}"
echo "Failed slopes    : ${FAILED_SLOPES[*]:-None}"
echo "Summary: total=$TOTAL_RUNS, success=${#SUCCESSFUL_SLOPES[@]}, failed=${#FAILED_SLOPES[@]}"

if (( ${#FAILED_SLOPES[@]} > 0 )); then
  echo "Completed with failures."
  exit 1
fi

echo "All surrogate-slope experiments completed successfully."
exit 0
