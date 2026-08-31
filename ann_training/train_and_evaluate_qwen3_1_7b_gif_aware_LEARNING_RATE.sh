#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_1_7b_tldr__gif_aware.yaml"

# Learning rates to run sequentially.
LEARNING_RATES=(
  2.0e-05
  1.0e-06
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ./ann_training/train_and_evaluate_qwen3_1_7b_gif_aware_LEARNING_RATE.sh
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
echo "Qwen3-1.7B GIF-Aware ANN Training + Evaluation"
echo "============================================================"
echo "Project root          : $PROJECT_ROOT"
echo "Source config         : $SOURCE_CFG"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                  : $NGPU"
echo "Learning rates        : ${LEARNING_RATES[*]}"
echo "Total LR runs         : ${#LEARNING_RATES[@]}"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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

override_learning_rate() {
  local run_cfg="$1"
  local learning_rate="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
learning_rate = sys.argv[2]

text = path.read_text(encoding="utf-8")

updated, count = re.subn(
    r"(?m)^(  learning_rate:)\s*\S+\s*$",
    rf"\1 {learning_rate}",
    text,
)

if count != 1:
    raise SystemExit(
        f"Expected exactly one training.learning_rate entry, found {count}"
    )

path.write_text(
    updated + ("" if updated.endswith("\n") else "\n"),
    encoding="utf-8",
)
' "$run_cfg" "$learning_rate"
}

assert_learning_rate() {
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

values = re.findall(
    r"(?m)^  learning_rate:\s*(\S+)\s*$",
    text,
)

if len(values) != 1:
    raise SystemExit(
        f"Expected exactly one training.learning_rate entry, found {len(values)}"
    )

actual = Decimal(values[0])

if actual != expected:
    raise SystemExit(
        f"training.learning_rate mismatch: "
        f"expected={expected}, actual={actual}"
    )

print(f"Confirmed training.learning_rate={values[0]} in {path}")
' "$run_cfg" "$learning_rate"
}

# ---------------------------------------------------------------------------
# Run one learning-rate experiment
# ---------------------------------------------------------------------------

run_one_learning_rate() {
  local learning_rate="$1"
  local normalized_learning_rate
  local lock_key
  local lock_file
  local run_cfg
  local train_status
  local eval_status

  echo
  echo "============================================================"
  echo "Starting learning_rate=$learning_rate"
  echo "GPUs=$CUDA_VISIBLE_DEVICES"
  echo "NGPU=$NGPU"
  echo "============================================================"

  # Normalize learning rate so that the lock filename is stable.
  if ! normalized_learning_rate="$(normalize_learning_rate "$learning_rate")"; then
    echo "[FAILED] Invalid learning rate: $learning_rate" >&2
    return 1
  fi

  lock_key="${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-qwen3-1.7b-gif-aware-lr-${lock_key}.lock"

  # Prevent concurrent writers for the same learning rate.
  exec 9>"$lock_file"

  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using learning_rate=$learning_rate" >&2
    exec 9>&-
    return 1
  fi

  # Create an independent config for this learning rate.
  run_cfg="$(mktemp "/tmp/snn-qwen3-1.7b-gif-aware-${lock_key}.XXXXXX.yaml")"

  echo "Per-instance config: $run_cfg"

  cleanup_run_cfg() {
    rm -f -- "$run_cfg"
  }

  # Copy the source config into the per-instance config.
  if ! cp -- "$SOURCE_CFG" "$run_cfg"; then
    echo "[FAILED] Failed to copy source config to per-instance config" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  # -------------------------------------------------------------------------
  # Configure and verify learning rate
  # -------------------------------------------------------------------------

  if ! override_learning_rate "$run_cfg" "$learning_rate"; then
    echo "[FAILED] Failed to override learning_rate=$learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  if ! assert_learning_rate "$run_cfg" "$learning_rate"; then
    echo "[FAILED] learning_rate verification failed for $learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  # -------------------------------------------------------------------------
  # ANN training
  # -------------------------------------------------------------------------

  echo
  echo "[TRAIN] learning_rate=$learning_rate"
  echo "[TRAIN] torchrun --nproc_per_node=$NGPU"

  set +e

  torchrun \
    --standalone \
    --nproc_per_node="$NGPU" \
    scripts/train_ann.py \
    --config "$run_cfg"

  train_status=$?

  set -e

  if ((train_status != 0)); then
    echo
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[FAILED] ANN training failed"
    echo "learning_rate=$learning_rate"
    echo "exit_code=$train_status"
    echo "Skipping evaluation and continuing to next learning rate."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return "$train_status"
  fi

  echo
  echo "[TRAIN] Completed successfully for learning_rate=$learning_rate"

  # -------------------------------------------------------------------------
  # Re-apply and verify learning rate before evaluation
  # -------------------------------------------------------------------------

  if ! override_learning_rate "$run_cfg" "$learning_rate"; then
    echo "[FAILED] Failed to re-apply learning_rate before evaluation" >&2

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return 1
  fi

  if ! assert_learning_rate "$run_cfg" "$learning_rate"; then
    echo "[FAILED] learning_rate verification failed before evaluation" >&2

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return 1
  fi

  # -------------------------------------------------------------------------
  # ANN evaluation
  # -------------------------------------------------------------------------

  echo
  echo "[EVAL] learning_rate=$learning_rate"
  echo "[EVAL] accelerate --num_processes $NGPU"

  set +e

  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$run_cfg" \
    --neuron ann

  eval_status=$?

  set -e

  if ((eval_status != 0)); then
    echo
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[FAILED] ANN evaluation failed"
    echo "learning_rate=$learning_rate"
    echo "exit_code=$eval_status"
    echo "Continuing to next learning rate."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return "$eval_status"
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[SUCCESS] learning_rate=$learning_rate"
  echo "------------------------------------------------------------"

  cleanup_run_cfg
  flock -u 9
  exec 9>&-

  return 0
}

# ---------------------------------------------------------------------------
# Run all learning rates sequentially
# ---------------------------------------------------------------------------

declare -a FAILED_LEARNING_RATES=()
declare -a SUCCESSFUL_LEARNING_RATES=()

TOTAL_RUNS="${#LEARNING_RATES[@]}"
CURRENT_RUN=0

for learning_rate in "${LEARNING_RATES[@]}"; do
  CURRENT_RUN=$((CURRENT_RUN + 1))

  echo
  echo "################################################################"
  echo "# Experiment $CURRENT_RUN / $TOTAL_RUNS"
  echo "# learning_rate=$learning_rate"
  echo "################################################################"

  if run_one_learning_rate "$learning_rate"; then
    SUCCESSFUL_LEARNING_RATES+=("$learning_rate")
  else
    FAILED_LEARNING_RATES+=("$learning_rate")

    echo
    echo "[WARNING] learning_rate=$learning_rate failed."
    echo "[WARNING] Continuing to the next learning rate..."
  fi
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo
echo
echo "================================================================"
echo "All learning-rate experiments have finished."
echo "================================================================"

echo
echo "Successful learning rates:"
if (( ${#SUCCESSFUL_LEARNING_RATES[@]} == 0 )); then
  echo "  None"
else
  for learning_rate in "${SUCCESSFUL_LEARNING_RATES[@]}"; do
    echo "  $learning_rate"
  done
fi

echo
echo "Failed learning rates:"
if (( ${#FAILED_LEARNING_RATES[@]} == 0 )); then
  echo "  None"
else
  for learning_rate in "${FAILED_LEARNING_RATES[@]}"; do
    echo "  $learning_rate"
  done
fi

echo
echo "Summary:"
echo "  Total    : $TOTAL_RUNS"
echo "  Success  : ${#SUCCESSFUL_LEARNING_RATES[@]}"
echo "  Failed   : ${#FAILED_LEARNING_RATES[@]}"

echo
echo "================================================================"

# Return non-zero only after ALL learning rates have been attempted.
# This means a failure in one LR never prevents subsequent LRs from running.
if (( ${#FAILED_LEARNING_RATES[@]} > 0 )); then
  echo "Completed with failures."
  exit 1
fi

echo "All learning-rate experiments completed successfully."
exit 0