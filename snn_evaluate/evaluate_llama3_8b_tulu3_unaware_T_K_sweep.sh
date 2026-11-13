#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
CFG="${CFG:-configs/generated/exp2_llama3_8b_tulu3__unaware.yaml}"
USE_POSTS=(true)

# Let PyTorch grow allocator segments during long multi-GPU lm-eval runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA_VISIBLE_DEVICES=1,2,3 ./snn_evaluate/evaluate_llama3_8b_tulu3_unaware_T_K_sweep.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set, for example: CUDA_VISIBLE_DEVICES=0,1 $0" >&2
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
  echo "Missing config: $CFG. Run python scripts/materialize_configs.py first." >&2
  exit 1
fi

run_evaluation() {
  local config_path="$1"
  shift
  echo "============================================================"
  echo "Running: accelerate launch --num_processes $NGPU scripts/evaluate_lm_harness.py --config $config_path $*"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  echo "============================================================"
  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_lm_harness.py \
    --config "$config_path" \
    "$@"
}

wait_for_gpu_release() {
  echo "Evaluation completed; waiting 5 seconds for GPU memory to be released..."
  sleep 5
}

for use_post in "${USE_POSTS[@]}"; do
  override_cfg="$(mktemp /tmp/llama3_8b_tulu3_unaware_use_post_XXXXXX.yaml)"
  trap 'rm -f "$override_cfg"' EXIT

  python - "$CFG" "$override_cfg" "$use_post" <<'PYCFG'
import sys
import yaml

source_path, target_path, value = sys.argv[1:]
with open(source_path, encoding="utf-8") as source:
    config = yaml.safe_load(source)
config["conversion"]["use_post_finetuning_artifacts"] = value == "true"
with open(target_path, "w", encoding="utf-8") as target:
    yaml.safe_dump(config, target, sort_keys=False)
PYCFG

  echo "============================================================"
  echo "Running Tulu-3 lm-eval SNN sweep with conversion.use_post_finetuning_artifacts=$use_post"
  echo "============================================================"

  # Conversion descriptors must exist before their corresponding SNN lm-eval.
  # python scripts/convert_snn.py --config "$override_cfg" --neuron gif
  python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 4
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 6

  # Optional larger deployment sweeps; uncomment conversion and evaluation together.
  # python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 6
  # python scripts/convert_snn.py --config "$override_cfg" --neuron phase --phase-T 8
  # python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 6
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 8
  python scripts/convert_snn.py --config "$override_cfg" --neuron mtn --mtn-T 8 --mtn-K 10

  # run_evaluation "$override_cfg" --neuron gif
  # wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron phase --phase-T 4
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 4 --mtn-K 6
  wait_for_gpu_release

  # run_evaluation "$override_cfg" --neuron phase --phase-T 6
  # wait_for_gpu_release
  # run_evaluation "$override_cfg" --neuron phase --phase-T 8
  # wait_for_gpu_release
  # run_evaluation "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 6
  # wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 6 --mtn-K 8
  wait_for_gpu_release
  run_evaluation "$override_cfg" --neuron mtn --mtn-T 8 --mtn-K 10
  wait_for_gpu_release

  rm -f "$override_cfg"
  trap - EXIT
done
