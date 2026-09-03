#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
CFG="${CFG:-configs/generated/exp1_qwen3_8b_tldr__gif_aware.yaml}"
USE_POSTS=(true false)

# CUDA_VISIBLE_DEVICES=0,1 ./snn_evaluate/evaluate_qwen3_8b_gif_aware_T_K_sweep.sh

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
  echo "Evaluation completed; waiting 30 seconds for GPU memory to be released..."
  sleep 30
}

# Step 7: generate the Post-finetuning Prefix for the final ANN checkpoint.
python scripts/discover_prefix.py \
  --config "$CFG" \
  --stage post_finetuning

# Step 8: create Post-finetuning conversion Stage A calibration artifacts.
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase A

for use_post in "${USE_POSTS[@]}"; do
  override_cfg="$(mktemp /tmp/qwen3_8b_gif_aware_use_post_XXXXXX.yaml)"
  trap 'rm -f "$override_cfg"' EXIT

  python - "$CFG" "$override_cfg" "$use_post" <<'PY'
import sys

import yaml

source_path, target_path, value = sys.argv[1:]
with open(source_path, encoding="utf-8") as source:
    config = yaml.safe_load(source)
config["conversion"]["use_post_finetuning_artifacts"] = value == "true"
with open(target_path, "w", encoding="utf-8") as target:
    yaml.safe_dump(config, target, sort_keys=False)
PY

  echo "============================================================"
  echo "Running sweep with conversion.use_post_finetuning_artifacts=$use_post"
  echo "============================================================"

  python scripts/convert_snn.py --config "$override_cfg" --neuron gif

  python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 4
  python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 6
  python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 8

  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 6
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 6
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 8
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 8 --mtn-K 10

  run_evaluation "$override_cfg" --neuron gif
  wait_for_gpu_release

  run_evaluation "$override_cfg" --neuron phase --phase-T 4
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron phase --phase-T 6
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron phase --phase-T 8
  wait_for_gpu_release

  run_evaluation "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 6
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 6
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 8
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 8 --mtn-K 10

  rm -f "$override_cfg"
  trap - EXIT
done
