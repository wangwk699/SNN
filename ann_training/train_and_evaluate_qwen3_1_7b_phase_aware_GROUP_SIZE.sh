#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml"

# Group sizes to run sequentially.
GROUP_SIZES=(
  -1
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ./ann_training/train_and_evaluate_qwen3_1_7b_phase_aware_GROUP_SIZE.sh
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
echo "Group sizes           : ${GROUP_SIZES[*]}"
echo "Total group-size runs : ${#GROUP_SIZES[@]}"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
validate_group_size() {
  if [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "GROUP_SIZE must be a positive integer: $1" >&2
    return 1
  fi

  printf '%s\n' "$1"
}

override_group_size() {
  local run_cfg="$1"
  local group_size="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
group_size = sys.argv[2]

text = path.read_text(encoding="utf-8")

updated, count = re.subn(
    r"(?m)^(  group_size:)[ \t]*\S+[ \t]*$",
    rf"\1 {group_size}",
    text,
)

if count != 1:
    raise SystemExit(
        f"Expected exactly one calibration.group_size entry, found {count}"
    )

path.write_text(
    updated + ("" if updated.endswith("\n") else "\n"),
    encoding="utf-8",
)
' "$run_cfg" "$group_size"
}

assert_group_size() {
  local run_cfg="$1"
  local group_size="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])

text = path.read_text(encoding="utf-8")

values = re.findall(
    r"(?m)^  group_size:[ \t]*(\S+)[ \t]*$",
    text,
)

if len(values) != 1:
    raise SystemExit(
        f"Expected exactly one calibration.group_size entry, found {len(values)}"
    )

actual = int(values[0])

if actual != expected:
    raise SystemExit(
        f"calibration.group_size mismatch: "
        f"expected={expected}, actual={actual}"
    )

print(f"Confirmed calibration.group_size={values[0]} in {path}")
' "$run_cfg" "$group_size"
}

# ---------------------------------------------------------------------------
# Run one group-size experiment
# ---------------------------------------------------------------------------

run_one_group_size() {
  local group_size="$1"
  local validated_group_size
  local lock_key
  local lock_file
  local run_cfg
  local train_status
  local eval_status

  echo
  echo "============================================================"
  echo "Starting group_size=$group_size"
  echo "GPUs=$CUDA_VISIBLE_DEVICES"
  echo "NGPU=$NGPU"
  echo "============================================================"

  # Validate group size and use it as a stable lock key.
  if ! validated_group_size="$(validate_group_size "$group_size")"; then
    echo "[FAILED] Invalid group size: $group_size" >&2
    return 1
  fi

  lock_key="${validated_group_size//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-qwen3-1.7b-phase-aware-group-size-${lock_key}.lock"

  # Prevent concurrent writers for the same group size.
  exec 9>"$lock_file"

  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using group_size=$group_size" >&2
    exec 9>&-
    return 1
  fi

  # Create an independent config for this group size.
  run_cfg="$(mktemp "/tmp/snn-qwen3-1.7b-phase-aware-group-size-${lock_key}.XXXXXX.yaml")"

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
  # Configure and verify group size
  # -------------------------------------------------------------------------

  if ! override_group_size "$run_cfg" "$group_size"; then
    echo "[FAILED] Failed to override group_size=$group_size" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  if ! assert_group_size "$run_cfg" "$group_size"; then
    echo "[FAILED] group_size verification failed for $group_size" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  # -------------------------------------------------------------------------
  # ANN training
  # -------------------------------------------------------------------------

  echo
  echo "[TRAIN] group_size=$group_size"
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
    echo "group_size=$group_size"
    echo "exit_code=$train_status"
    echo "Skipping evaluation and continuing to next group size."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return "$train_status"
  fi

  echo
  echo "[TRAIN] Completed successfully for group_size=$group_size"

  # -------------------------------------------------------------------------
  # Re-apply and verify group size before evaluation
  # -------------------------------------------------------------------------

  if ! override_group_size "$run_cfg" "$group_size"; then
    echo "[FAILED] Failed to re-apply group_size before evaluation" >&2

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return 1
  fi

  if ! assert_group_size "$run_cfg" "$group_size"; then
    echo "[FAILED] group_size verification failed before evaluation" >&2

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return 1
  fi

  # -------------------------------------------------------------------------
  # ANN evaluation
  # -------------------------------------------------------------------------

  echo
  echo "[EVAL] group_size=$group_size"
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
    echo "group_size=$group_size"
    echo "exit_code=$eval_status"
    echo "Continuing to next group size."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    cleanup_run_cfg
    flock -u 9
    exec 9>&-

    return "$eval_status"
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[SUCCESS] group_size=$group_size"
  echo "------------------------------------------------------------"

  cleanup_run_cfg
  flock -u 9
  exec 9>&-

  return 0
}

# ---------------------------------------------------------------------------
# Run all group sizes sequentially
# ---------------------------------------------------------------------------

declare -a FAILED_GROUP_SIZES=()
declare -a SUCCESSFUL_GROUP_SIZES=()

TOTAL_RUNS="${#GROUP_SIZES[@]}"
CURRENT_RUN=0

for group_size in "${GROUP_SIZES[@]}"; do
  CURRENT_RUN=$((CURRENT_RUN + 1))

  echo
  echo "################################################################"
  echo "# Experiment $CURRENT_RUN / $TOTAL_RUNS"
  echo "# group_size=$group_size"
  echo "################################################################"

  if run_one_group_size "$group_size"; then
    SUCCESSFUL_GROUP_SIZES+=("$group_size")
  else
    FAILED_GROUP_SIZES+=("$group_size")

    echo
    echo "[WARNING] group_size=$group_size failed."
    echo "[WARNING] Continuing to the next group size..."
  fi
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo
echo
echo "================================================================"
echo "All group-size experiments have finished."
echo "================================================================"

echo
echo "Successful group sizes:"
if (( ${#SUCCESSFUL_GROUP_SIZES[@]} == 0 )); then
  echo "  None"
else
  for group_size in "${SUCCESSFUL_GROUP_SIZES[@]}"; do
    echo "  $group_size"
  done
fi

echo
echo "Failed group sizes:"
if (( ${#FAILED_GROUP_SIZES[@]} == 0 )); then
  echo "  None"
else
  for group_size in "${FAILED_GROUP_SIZES[@]}"; do
    echo "  $group_size"
  done
fi

echo
echo "Summary:"
echo "  Total    : $TOTAL_RUNS"
echo "  Success  : ${#SUCCESSFUL_GROUP_SIZES[@]}"
echo "  Failed   : ${#FAILED_GROUP_SIZES[@]}"

echo
echo "================================================================"

# Return non-zero only after ALL group sizes have been attempted.
# This means a failure for one group size never prevents subsequent runs.
if (( ${#FAILED_GROUP_SIZES[@]} > 0 )); then
  echo "Completed with failures."
  exit 1
fi

echo "All group-size experiments completed successfully."
exit 0