#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml"

# Learning rates to run sequentially.
LEARNING_RATES=(
  1.0e-05
  2.0e-05
  5.0e-05
  8.0e-05
)

PHASE_PREVIOUS_LAYERS_SNN_VALUES=(
  false
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,3,7 ./ann_training/qwen3_1_7b/train_and_evaluate_qwen3_1_7b_phase_aware_LEARNING_RATE.sh
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
echo "Qwen3-1.7B Phase-Aware ANN Training + Evaluation"
echo "============================================================"
echo "Project root          : $PROJECT_ROOT"
echo "Source config         : $SOURCE_CFG"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                  : $NGPU"
echo "Learning rates        : ${LEARNING_RATES[*]}"
echo "Previous-layers SNN  : ${PHASE_PREVIOUS_LAYERS_SNN_VALUES[*]}"
echo "Total combinations   : $((${#PHASE_PREVIOUS_LAYERS_SNN_VALUES[@]} * ${#LEARNING_RATES[@]}))"
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

override_phase_previous_layers_snn() {
  local run_cfg="$1"
  local value="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
value = sys.argv[2]
if value not in {"true", "false"}:
    raise SystemExit("calibration.phase_previous_layers_snn must be true or false")

text = path.read_text(encoding="utf-8")
updated, count = re.subn(
    r"(?m)^(  phase_previous_layers_snn:)\s*\S+\s*$",
    rf"\1 {value}",
    text,
)
if count != 1:
    raise SystemExit(
        f"Expected exactly one calibration.phase_previous_layers_snn entry, found {count}"
    )
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
' "$run_cfg" "$value"
}

assert_phase_previous_layers_snn() {
  local run_cfg="$1"
  local expected="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
values = re.findall(
    r"(?m)^  phase_previous_layers_snn:\s*(\S+)\s*$",
    path.read_text(encoding="utf-8"),
)
if len(values) != 1:
    raise SystemExit(
        f"Expected exactly one calibration.phase_previous_layers_snn entry, found {len(values)}"
    )
if values[0] != expected:
    raise SystemExit(
        f"calibration.phase_previous_layers_snn mismatch: expected={expected}, actual={values[0]}"
    )
print(f"Confirmed calibration.phase_previous_layers_snn={values[0]} in {path}")
' "$run_cfg" "$expected"
}

configure_run() {
  local run_cfg="$1"
  local previous_layers_snn="$2"
  local learning_rate="$3"

  override_phase_previous_layers_snn "$run_cfg" "$previous_layers_snn" \
    && assert_phase_previous_layers_snn "$run_cfg" "$previous_layers_snn" \
    && override_learning_rate "$run_cfg" "$learning_rate" \
    && assert_learning_rate "$run_cfg" "$learning_rate"
}

# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

run_one_learning_rate() {
  local previous_layers_snn="$1"
  local learning_rate="$2"
  local normalized_learning_rate
  local lock_key
  local lock_file
  local run_cfg
  local train_status
  local eval_status

  echo
  echo "============================================================"
  echo "Starting phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
  echo "GPUs=$CUDA_VISIBLE_DEVICES"
  echo "NGPU=$NGPU"
  echo "============================================================"

  # Normalize learning rate so that the lock filename is stable.
  if ! normalized_learning_rate="$(normalize_learning_rate "$learning_rate")"; then
    echo "[FAILED] Invalid learning rate: $learning_rate" >&2
    return 1
  fi

  lock_key="${previous_layers_snn}-${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-qwen3-1.7b-phase-aware-lr-${lock_key}.lock"

  # Prevent concurrent writers for the same configuration combination.
  exec 9>"$lock_file"

  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate" >&2
    exec 9>&-
    return 1
  fi

  # Create an independent config for this configuration combination.
  run_cfg="$(mktemp "/tmp/snn-qwen3-1.7b-phase-aware-${lock_key}.XXXXXX.yaml")"

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
  # Configure and verify phase_previous_layers_snn and learning rate
  # -------------------------------------------------------------------------

  if ! configure_run "$run_cfg" "$previous_layers_snn" "$learning_rate"; then
    echo "[FAILED] Configuration failed: phase_previous_layers_snn=$previous_layers_snn, learning_rate=$learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  # -------------------------------------------------------------------------
  # ANN training
  # -------------------------------------------------------------------------

  echo
  echo "[TRAIN] phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
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
    echo "phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
    echo "exit_code=$train_status"
    echo "Skipping evaluation and continuing to next combination."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return "$train_status"
  fi

  echo
  echo "[TRAIN] Completed successfully for phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"

  # -------------------------------------------------------------------------
  # Re-apply and verify phase_previous_layers_snn and learning rate before evaluation
  # -------------------------------------------------------------------------

  if ! configure_run "$run_cfg" "$previous_layers_snn" "$learning_rate"; then
    echo "[FAILED] Configuration failed before evaluation: phase_previous_layers_snn=$previous_layers_snn, learning_rate=$learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  # -------------------------------------------------------------------------
  # ANN evaluation
  # -------------------------------------------------------------------------

  echo
  echo "[EVAL] phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
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
    echo "phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
    echo "exit_code=$eval_status"
    echo "Continuing to next combination."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return "$eval_status"
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[SUCCESS] phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
  echo "------------------------------------------------------------"

  cleanup_run_cfg
  flock -u 9
  exec 9>&-

  return 0
}

# ---------------------------------------------------------------------------
# Run all configuration combinations sequentially
# ---------------------------------------------------------------------------

declare -a FAILED_COMBINATIONS=()
declare -a SUCCESSFUL_COMBINATIONS=()

TOTAL_RUNS="$((${#PHASE_PREVIOUS_LAYERS_SNN_VALUES[@]} * ${#LEARNING_RATES[@]}))"
CURRENT_RUN=0

for previous_layers_snn in "${PHASE_PREVIOUS_LAYERS_SNN_VALUES[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))

    echo
    echo "################################################################"
    echo "# Experiment $CURRENT_RUN / $TOTAL_RUNS"
    echo "# phase_previous_layers_snn=$previous_layers_snn learning_rate=$learning_rate"
    echo "################################################################"

    combination="phase_previous_layers_snn=$previous_layers_snn,learning_rate=$learning_rate"
    if run_one_learning_rate "$previous_layers_snn" "$learning_rate"; then
      SUCCESSFUL_COMBINATIONS+=("$combination")
    else
      FAILED_COMBINATIONS+=("$combination")

      echo
      echo "[WARNING] $combination failed."
      echo "[WARNING] Continuing to the next combination..."
    fi
  done
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo
echo
echo "================================================================"
echo "All configuration combinations have finished."
echo "================================================================"

echo
echo "Successful combinations:"
if (( ${#SUCCESSFUL_COMBINATIONS[@]} == 0 )); then
  echo "  None"
else
  for combination in "${SUCCESSFUL_COMBINATIONS[@]}"; do
    echo "  $combination"
  done
fi

echo
echo "Failed combinations:"
if (( ${#FAILED_COMBINATIONS[@]} == 0 )); then
  echo "  None"
else
  for combination in "${FAILED_COMBINATIONS[@]}"; do
    echo "  $combination"
  done
fi

echo
echo "Summary:"
echo "  Total    : $TOTAL_RUNS"
echo "  Success  : ${#SUCCESSFUL_COMBINATIONS[@]}"
echo "  Failed   : ${#FAILED_COMBINATIONS[@]}"

echo
echo "================================================================"

# Return non-zero only after ALL combinations have been attempted.
# A failed combination does not prevent subsequent combinations from running.
if (( ${#FAILED_COMBINATIONS[@]} > 0 )); then
  echo "Completed with failures."
  exit 1
fi

echo "All configuration combinations completed successfully."
exit 0