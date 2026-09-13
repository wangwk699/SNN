#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml"
REQUIRED_GPUS="0,1,2"
NGPU=3

# CUDA_VISIBLE_DEVICES is supplied by the caller.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 \
#     ./ann_training/llama3_8b/run_tulu3_gif_aware_tuning_v2.sh
gpu_devices="${CUDA_VISIBLE_DEVICES:-}"

cd "$PROJECT_ROOT"

if [[ ! -f "$SOURCE_CFG" ]]; then
  echo "Missing generated config: $PROJECT_ROOT/$SOURCE_CFG" >&2
  exit 1
fi
if [[ "$gpu_devices" != "$REQUIRED_GPUS" ]]; then
  echo "This v2 plan requires CUDA_VISIBLE_DEVICES=$REQUIRED_GPUS; got '${gpu_devices:-unset}'." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu_devices"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec 9>"/tmp/snn-tulu3-gif-aware-tuning-v2.lock"
if ! flock -n 9; then
  echo "Another Tulu-3 GIF-aware tuning-v2 driver is already running." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU occupancy preflight." >&2
  exit 1
fi

gpu_busy=0
IFS=',' read -r -a gpu_list <<< "$gpu_devices"
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

echo "Tulu-3 GIF-aware formal calibration/ratio tuning v2"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "CUDA allocator       : $PYTORCH_CUDA_ALLOC_CONF"
echo "Source config        : $SOURCE_CFG (read-only)"
echo "Candidates           : 3 calibration sizes x 4 GIF ratio pairs = 12"
echo "Calibration samples  : 128 512 1024"
echo "GIF ratio pairs      : (0.9,0.1) (0.8,0.2) (0.7,0.3) (0.5,0.5)"
echo "Formal train samples : 10000"
echo "Fixed optimizer      : lr=5e-6 cosine warmup=0.01 epochs=1 grad_accum=16"
echo "Fixed semantics      : phase_T=4 mtn_T=4 group_size=128 hard_clip/hard_clip"
echo "Memory checkpoints   : attention_core=true mlp=true"
echo "Per candidate        : manifest/Prefix -> Calibration A/B -> train -> six-task eval"
echo "Failure policy       : record failure and continue to the next candidate"

python3 scripts/run_tulu3_gif_aware_tuning_v2.py \
  --config "$SOURCE_CFG" \
  --num-processes "$NGPU"
