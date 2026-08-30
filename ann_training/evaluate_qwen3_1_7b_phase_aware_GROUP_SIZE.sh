#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml"

GROUP_SIZES=(
  128
  32
)

# Respect an externally supplied device list; default to the requested GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

cd "$PROJECT_ROOT"

if [[ ! -f "$SOURCE_CFG" ]]; then
  echo "Missing generated config: $PROJECT_ROOT/$SOURCE_CFG" >&2
  exit 1
fi

IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
NGPU="${#gpu_list[@]}"
if (( NGPU <= 0 )); then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU ID" >&2
  exit 1
fi

for gpu in "${gpu_list[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID in CUDA_VISIBLE_DEVICES: $gpu" >&2
    exit 1
  fi
done

for group_size in "${GROUP_SIZES[@]}"; do
  if [[ ! "$group_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid calibration.group_size: $group_size" >&2
    exit 1
  fi

  run_cfg="$(mktemp "/tmp/snn-qwen3-1.7b-phase-aware-eval-group-size-${group_size}.XXXXXX.yaml")"

  cleanup_run_cfg() {
    rm -f -- "$run_cfg"
  }
  trap cleanup_run_cfg EXIT

  cp -- "$SOURCE_CFG" "$run_cfg"

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

  CFG_17_P="$run_cfg"

  echo "============================================================"
  echo "Evaluating calibration.group_size=$group_size"
  echo "Config: $CFG_17_P"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "Number of processes: $NGPU"
  echo "============================================================"

  for NEURON in phase gif mtn; do
    accelerate launch --num_processes "$NGPU" \
      scripts/evaluate_tldr.py \
      --config "$CFG_17_P" \
      --neuron "$NEURON"
  done

  cleanup_run_cfg
  trap - EXIT
done
