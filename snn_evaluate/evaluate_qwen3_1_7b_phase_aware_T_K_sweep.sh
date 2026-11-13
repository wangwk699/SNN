#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
CFG="${CFG:-configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml}"
LEARNING_RATES=(
  2.0e-05
)

# CUDA_VISIBLE_DEVICES=0,3 ./snn_evaluate/evaluate_qwen3_1_7b_phase_aware_T_K_sweep.sh

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
  echo "Running: accelerate launch --num_processes $NGPU scripts/evaluate_tldr.py --config $1 ${*:2}"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "============================================================"

  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$1" \
    "${@:2}"
}

wait_for_gpu_release() {
  echo "Evaluation completed; waiting 10 seconds for GPU memory to be released..."
  sleep 10
}

for learning_rate in "${LEARNING_RATES[@]}"; do
  override_cfg="$(mktemp "/tmp/qwen3_1_7b_phase_aware_lr_${learning_rate}_XXXXXX.yaml")"
  trap 'rm -f "$override_cfg"' EXIT

  python - "$CFG" "$override_cfg" "$learning_rate" <<'PY'
import sys

import yaml

source_path, target_path, learning_rate = sys.argv[1:]
with open(source_path, encoding="utf-8") as source:
    config = yaml.safe_load(source)
config["training"]["learning_rate"] = float(learning_rate)
with open(target_path, "w", encoding="utf-8") as target:
    yaml.safe_dump(config, target, sort_keys=False)
PY

  echo "============================================================"
  echo "Running sweep with training.learning_rate=$learning_rate"
  echo "============================================================"

  python scripts/convert_snn.py --config "$override_cfg" --neuron gif

  python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 4
  # python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 6
  # python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 8

  # python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 4
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 6
  # python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 6
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 8
  # python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 8 --mtn-K 10

  run_evaluation "$override_cfg" --neuron gif
  wait_for_gpu_release

  run_evaluation "$override_cfg" --neuron phase --phase-T 4
  wait_for_gpu_release
  # run_evaluation "$override_cfg" --neuron phase --phase-T 6
  # wait_for_gpu_release
  # run_evaluation "$override_cfg" --neuron phase --phase-T 8
  # wait_for_gpu_release

  # run_evaluation "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 4
  # wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 6
  wait_for_gpu_release
  # run_evaluation "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 6
  # wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 8
  # wait_for_gpu_release
  # run_evaluation "$override_cfg" --neuron mtn --mtn-T 8 --mtn-K 10

  rm -f "$override_cfg"
  trap - EXIT
done
