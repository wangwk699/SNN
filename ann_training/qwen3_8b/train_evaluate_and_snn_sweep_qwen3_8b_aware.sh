#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
CFG="${CFG:-configs/generated/exp1_qwen3_8b_tldr__gif_aware.yaml}"
USE_POSTS=(true false)

# Example:
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# CFG=configs/other.yaml \
# ./ann_training/qwen3_8b/train_evaluate_and_snn_sweep_qwen3_8b_gif_aware.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set, for example: CUDA_VISIBLE_DEVICES=4,5,6,7 $0" >&2
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

TRAIN_CFG=""
SNN_STAGE_CFG=""
SWEEP_CFG=""

cleanup() {
  rm -f "${TRAIN_CFG:-}" "${SNN_STAGE_CFG:-}" "${SWEEP_CFG:-}"
}
trap cleanup EXIT

write_config_override() {
  local source_cfg="$1"
  local target_cfg="$2"
  local test_samples="$3"
  local use_post="${4:-}"

  python - "$source_cfg" "$target_cfg" "$test_samples" "$use_post" <<'PY'
import sys

import yaml

source_path, target_path, test_samples, use_post = sys.argv[1:]
with open(source_path, encoding="utf-8") as source:
    config = yaml.safe_load(source)

config["evaluation"]["tldr_test_samples"] = int(test_samples)
if use_post:
    config["conversion"]["use_post_finetuning_artifacts"] = use_post == "true"

with open(target_path, "w", encoding="utf-8") as target:
    yaml.safe_dump(config, target, sort_keys=False)
PY
}

run_evaluation() {
  local evaluation_cfg="$1"
  shift

  echo "============================================================"
  echo "Running: accelerate launch --num_processes $NGPU scripts/evaluate_tldr.py --config $evaluation_cfg $*"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "============================================================"

  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$evaluation_cfg" \
    "$@"
}

wait_for_gpu_release() {
  echo "Evaluation completed; waiting 30 seconds for GPU memory to be released..."
  sleep 30
}

echo "============================================================"
echo "Qwen3-8B GIF-aware training, ANN evaluation, and SNN sweep"
echo "Base config          : $CFG"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "============================================================"

# Train and evaluate the final ANN with 1,000 TL;DR test samples.
TRAIN_CFG="$(mktemp /tmp/qwen3_8b_gif_aware_train_XXXXXX.yaml)"
write_config_override "$CFG" "$TRAIN_CFG" 1000

echo "============================================================"
echo "Step 6: train final ANN checkpoint (evaluation.tldr_test_samples=1000)"
echo "============================================================"
torchrun \
  --standalone \
  --nproc_per_node="$NGPU" \
  scripts/train_ann.py \
  --config "$TRAIN_CFG"

echo "============================================================"
echo "Step 9: evaluate final ANN checkpoint (evaluation.tldr_test_samples=1000)"
echo "============================================================"
run_evaluation "$TRAIN_CFG" --neuron ann
wait_for_gpu_release

# All Post-finetuning and SNN evaluations use 128 TL;DR test samples.
SNN_STAGE_CFG="$(mktemp /tmp/qwen3_8b_gif_aware_snn_stage_XXXXXX.yaml)"
write_config_override "$CFG" "$SNN_STAGE_CFG" 128

echo "============================================================"
echo "Step 7: generate Post-finetuning Prefix"
echo "============================================================"
python scripts/discover_prefix.py \
  --config "$SNN_STAGE_CFG" \
  --stage post_finetuning

echo "============================================================"
echo "Step 8: generate Post-finetuning conversion Stage A"
echo "============================================================"
python scripts/calibrate_sites.py \
  --config "$SNN_STAGE_CFG" \
  --stage post_finetuning \
  --calibration-phase A

for use_post in "${USE_POSTS[@]}"; do
  SWEEP_CFG="$(mktemp /tmp/qwen3_8b_gif_aware_snn_use_post_XXXXXX.yaml)"
  write_config_override "$CFG" "$SWEEP_CFG" 128 "$use_post"

  echo "============================================================"
  echo "Step 10: SNN sweep with conversion.use_post_finetuning_artifacts=$use_post"
  echo "evaluation.tldr_test_samples=128"
  echo "============================================================"

  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron gif

  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron phase --phase-T 4
  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron phase --phase-T 6
  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron phase --phase-T 8

  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron mtn --mtn-T 4 --mtn-K 6
  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron mtn --mtn-T 6 --mtn-K 6
  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron mtn --mtn-T 6 --mtn-K 8
  python scripts/convert_snn.py --config "$SWEEP_CFG" --neuron mtn --mtn-T 8 --mtn-K 10

  run_evaluation "$SWEEP_CFG" --neuron gif
  wait_for_gpu_release

  run_evaluation "$SWEEP_CFG" --neuron phase --phase-T 4
  wait_for_gpu_release
  run_evaluation "$SWEEP_CFG" --neuron phase --phase-T 6
  wait_for_gpu_release
  run_evaluation "$SWEEP_CFG" --neuron phase --phase-T 8
  wait_for_gpu_release

  run_evaluation "$SWEEP_CFG" --neuron mtn --mtn-T 4 --mtn-K 6
  wait_for_gpu_release
  run_evaluation "$SWEEP_CFG" --neuron mtn --mtn-T 6 --mtn-K 6
  wait_for_gpu_release
  run_evaluation "$SWEEP_CFG" --neuron mtn --mtn-T 6 --mtn-K 8
  wait_for_gpu_release
  run_evaluation "$SWEEP_CFG" --neuron mtn --mtn-T 8 --mtn-K 10

  rm -f "$SWEEP_CFG"
  SWEEP_CFG=""
done
