#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml"

# Surrogate slopes to run sequentially.
SLOPES=(
  2.0
  4.0
  6.0
  8.0
  10.0
)

# Run each slope for both calibration trajectories.
PHASE_PREVIOUS_LAYERS_SNN_VALUES=(
  true
  false
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=1,2,3,5 ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_phase_aware_SURROGATE_SLOPE.sh
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
echo "Llama-3-8B Tulu-3 Phase-Aware ANN Training + lm-eval"
echo "============================================================"
echo "Project root         : $PROJECT_ROOT"
echo "Source config        : $SOURCE_CFG"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Previous-layers SNN  : ${PHASE_PREVIOUS_LAYERS_SNN_VALUES[*]}"
echo "Surrogate slopes     : ${SLOPES[*]}"
echo "Total combinations   : $((${#PHASE_PREVIOUS_LAYERS_SNN_VALUES[@]} * ${#SLOPES[@]}))"
echo "============================================================"
echo

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
    rf"\1 {slope}",
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

override_phase_previous_layers_snn() {
  local run_cfg="$1"
  local phase_previous_layers_snn="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
value = sys.argv[2]
if value not in {"true", "false"}:
    raise SystemExit(
        "calibration.phase_previous_layers_snn must be true or false"
    )

text = path.read_text(encoding="utf-8")
updated, count = re.subn(
    r"(?m)^(  phase_previous_layers_snn:)\s*\S+\s*$",
    rf"\1 {value}",
    text,
)
if count != 1:
    raise SystemExit(
        "Expected exactly one calibration.phase_previous_layers_snn entry, "
        f"found {count}"
    )

path.write_text(
    updated + ("" if updated.endswith("\n") else "\n"),
    encoding="utf-8",
)
' "$run_cfg" "$phase_previous_layers_snn"
}

assert_phase_previous_layers_snn() {
  local run_cfg="$1"
  local phase_previous_layers_snn="$2"

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
        "Expected exactly one calibration.phase_previous_layers_snn entry, "
        f"found {len(values)}"
    )
if values[0] != expected:
    raise SystemExit(
        "calibration.phase_previous_layers_snn mismatch: "
        f"expected={expected}, actual={values[0]}"
    )

print(
    "Confirmed calibration.phase_previous_layers_snn="
    f"{values[0]} in {path}"
)
' "$run_cfg" "$phase_previous_layers_snn"
}

configure_run() {
  local run_cfg="$1"
  local phase_previous_layers_snn="$2"
  local slope="$3"

  override_phase_previous_layers_snn \
    "$run_cfg" "$phase_previous_layers_snn" \
    && assert_phase_previous_layers_snn \
      "$run_cfg" "$phase_previous_layers_snn" \
    && override_slope "$run_cfg" "$slope" \
    && assert_slope "$run_cfg" "$slope"
}

run_one_slope() {
  local phase_previous_layers_snn="$1"
  local slope="$2"
  local normalized_slope
  local lock_key
  local lock_file
  local run_cfg
  local train_status
  local eval_status

  echo
  echo "============================================================"
  echo "Starting phase_previous_layers_snn=$phase_previous_layers_snn"
  echo "Starting surrogate_slope=$slope"
  echo "GPUs=$CUDA_VISIBLE_DEVICES"
  echo "NGPU=$NGPU"
  echo "============================================================"

  if ! normalized_slope="$(normalize_slope "$slope")"; then
    echo "[FAILED] Invalid surrogate slope: $slope" >&2
    return 1
  fi

  lock_key="${phase_previous_layers_snn}-${normalized_slope//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-llama3-8b-tulu3-phase-aware-slope-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using surrogate_slope=$slope" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-llama3-8b-tulu3-phase-aware-slope-${lock_key}.XXXXXX.yaml")"
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

  if ! configure_run "$run_cfg" "$phase_previous_layers_snn" "$slope"; then
    echo "[FAILED] configuration failed: phase_previous_layers_snn=$phase_previous_layers_snn, surrogate_slope=$slope" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo
  echo "[TRAIN] phase_previous_layers_snn=$phase_previous_layers_snn surrogate_slope=$slope"
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
    echo "[FAILED] ANN training failed: phase_previous_layers_snn=$phase_previous_layers_snn, surrogate_slope=$slope, exit_code=$train_status" >&2
    echo "Skipping evaluation and continuing to next surrogate slope." >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  echo "[TRAIN] Completed successfully for phase_previous_layers_snn=$phase_previous_layers_snn surrogate_slope=$slope"

  if ! configure_run "$run_cfg" "$phase_previous_layers_snn" "$slope"; then
    echo "[FAILED] configuration verification failed before evaluation" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo
  echo "[EVAL] phase_previous_layers_snn=$phase_previous_layers_snn surrogate_slope=$slope"
  echo "[EVAL] accelerate --num_processes $NGPU"

  set +e
  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_lm_harness.py \
    --config "$run_cfg" \
    --neuron ann
  eval_status=$?
  set -e

  if ((eval_status != 0)); then
    echo "[FAILED] ANN lm-eval failed: phase_previous_layers_snn=$phase_previous_layers_snn, surrogate_slope=$slope, exit_code=$eval_status" >&2
    echo "Continuing to next surrogate slope." >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$eval_status"
  fi

  echo "[SUCCESS] phase_previous_layers_snn=$phase_previous_layers_snn surrogate_slope=$slope"
  cleanup_run_cfg
  flock -u 9
  exec 9>&-
  return 0
}

declare -a FAILED_COMBINATIONS=()
declare -a SUCCESSFUL_COMBINATIONS=()
declare -a SKIPPED_COMBINATIONS=()
TOTAL_COMBINATIONS=$((${#PHASE_PREVIOUS_LAYERS_SNN_VALUES[@]} * ${#SLOPES[@]}))
CURRENT_RUN=0

for phase_previous_layers_snn in "${PHASE_PREVIOUS_LAYERS_SNN_VALUES[@]}"; do
  for slope in "${SLOPES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    combination="phase_previous_layers_snn=$phase_previous_layers_snn,surrogate_slope=$slope"

    echo
    echo "################################################################"
    echo "# Experiment $CURRENT_RUN / $TOTAL_COMBINATIONS"
    echo "# $combination"
    echo "################################################################"

    if [[ "$phase_previous_layers_snn" == "false" && "$slope" == "4.0" ]]; then
      echo "[SKIPPED] $combination has already been completed."
      SKIPPED_COMBINATIONS+=("$combination")
      continue
    fi

    if run_one_slope "$phase_previous_layers_snn" "$slope"; then
      SUCCESSFUL_COMBINATIONS+=("$combination")
    else
      FAILED_COMBINATIONS+=("$combination")
      echo "[WARNING] $combination failed; continuing to the next combination." >&2
    fi
  done
done

echo
echo "================================================================"
echo "All surrogate-slope experiments have finished."
echo "================================================================"
printf 'Successful combinations:\n  %s\n' "${SUCCESSFUL_COMBINATIONS[*]:-None}"
printf 'Skipped combinations:\n  %s\n' "${SKIPPED_COMBINATIONS[*]:-None}"
printf 'Failed combinations:\n  %s\n' "${FAILED_COMBINATIONS[*]:-None}"
echo "Summary: total=$TOTAL_COMBINATIONS, success=${#SUCCESSFUL_COMBINATIONS[@]}, skipped=${#SKIPPED_COMBINATIONS[@]}, failed=${#FAILED_COMBINATIONS[@]}"

if (( ${#FAILED_COMBINATIONS[@]} > 0 )); then
  echo "Completed with failures."
  exit 1
fi

echo "All non-skipped parameter combinations completed successfully."
exit 0
