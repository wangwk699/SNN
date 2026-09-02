#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
CFG="${CFG:-configs/generated/exp1_qwen3_1_7b_tldr__gif_aware.yaml}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set, for example: CUDA_VISIBLE_DEVICES=6,7 $0" >&2
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

cd "$PROJECT_ROOT"

if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi

run_evaluation() {
  echo "============================================================"
  echo "Running: accelerate launch --num_processes $NGPU scripts/evaluate_tldr.py --config $CFG $*"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "============================================================"

  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$CFG" \
    "$@"
}

wait_for_gpu_release() {
  echo "Evaluation completed; waiting 30 seconds for GPU memory to be released..."
  sleep 30
}

python scripts/convert_snn.py --config "$CFG" --neuron gif

python scripts/convert_snn.py --config "$CFG" --neuron phase --phase-T 4
python scripts/convert_snn.py --config "$CFG" --neuron phase --phase-T 6
python scripts/convert_snn.py --config "$CFG" --neuron phase --phase-T 8

python scripts/convert_snn.py --config "$CFG" --neuron mtn --mtn-T 4 --mtn-K 6
python scripts/convert_snn.py --config "$CFG" --neuron mtn --mtn-T 6 --mtn-K 6
python scripts/convert_snn.py --config "$CFG" --neuron mtn --mtn-T 6 --mtn-K 8
python scripts/convert_snn.py --config "$CFG" --neuron mtn --mtn-T 8 --mtn-K 10

run_evaluation --neuron gif
wait_for_gpu_release

run_evaluation --neuron phase --phase-T 4
wait_for_gpu_release
run_evaluation --neuron phase --phase-T 6
wait_for_gpu_release
run_evaluation --neuron phase --phase-T 8
wait_for_gpu_release

run_evaluation --neuron mtn --mtn-T 4 --mtn-K 6
wait_for_gpu_release
run_evaluation --neuron mtn --mtn-T 6 --mtn-K 6
wait_for_gpu_release
run_evaluation --neuron mtn --mtn-T 6 --mtn-K 8
wait_for_gpu_release
run_evaluation --neuron mtn --mtn-T 8 --mtn-K 10
