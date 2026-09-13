#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml"

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2 ./ann_training/llama3_8b/run_tulu3_gif_aware_tuning_v1.sh
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec 9>"/tmp/snn-tulu3-gif-aware-tuning-v1.lock"
if ! flock -n 9; then
  echo "Another Tulu-3 GIF-aware tuning-v1 driver is already running." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU occupancy preflight." >&2
  exit 1
fi
gpu_busy=0
for gpu in "${gpu_list[@]}"; do
  gpu_processes="$(nvidia-smi --id="$gpu" \
    --query-compute-apps=pid,used_memory,process_name \
    --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$gpu_processes" ]]; then
    echo "[BUSY] GPU $gpu already has compute processes:" >&2
    while IFS= read -r process_line; do
      echo "  $process_line" >&2
    done <<< "$gpu_processes"
    gpu_busy=1
  fi
done
if (( gpu_busy != 0 )); then
  echo "Refusing to overlap another job on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "Tulu-3 GIF-aware adaptive ANN tuning v1"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "CUDA allocator       : $PYTORCH_CUDA_ALLOC_CONF"
echo "Source config        : $SOURCE_CFG"
echo "Formal train samples : 10000"
echo "Fixed                : phase_T=4 mtn_T=4 group_size=128 grad_accum=16"
echo "Stages               : LR -> scheduler/warmup -> backward policy -> epochs/grad clip"
echo "Resume               : completed six-task results are reused"

python3 scripts/run_tulu3_gif_aware_tuning.py \
  --config "$SOURCE_CFG" \
  --num-processes "$NGPU"
