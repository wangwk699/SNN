#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"

CFG_8_V="configs/generated/exp1_qwen3_8b_tldr__vanilla.yaml"
CFG_8_U="configs/generated/exp1_qwen3_8b_tldr__unaware.yaml"
ALL_CFGS=("$CFG_8_V" "$CFG_8_U")

LEARNING_RATES=(
  1.0e-06
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#     ./ann_training/train_discover_prefix_and_evaluate_qwen3_8b_tldr_vanilla_unaware_LEARNING_RATE.sh
gpu_devices="${CUDA_VISIBLE_DEVICES:-}"

cd "$PROJECT_ROOT"

if [[ -z "$gpu_devices" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be provided and non-empty" >&2
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

for cfg in "${ALL_CFGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Missing generated config: $PROJECT_ROOT/$cfg" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$gpu_devices"

normalize_learning_rate() {
  python3 -c '
from decimal import Decimal, InvalidOperation
import sys

try:
    value = Decimal(sys.argv[1])
except InvalidOperation as exc:
    raise SystemExit(f"Invalid LEARNING_RATE: {sys.argv[1]}") from exc

if not value.is_finite() or value <= 0:
    raise SystemExit("LEARNING_RATE must be a positive finite number")

print(format(value.normalize(), "E"))
' "$1"
}

override_learning_rate() {
  local run_cfg="$1"
  local learning_rate="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
learning_rate = sys.argv[2]
text = path.read_text(encoding="utf-8")

updated, count = re.subn(
    r"(?m)^(  learning_rate:)\s*\S+\s*$",
    rf"\1 {learning_rate}",
    text,
)

if count != 1:
    raise SystemExit(
        f"Expected exactly one training.learning_rate entry, found {count}"
    )

path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
' "$run_cfg" "$learning_rate"
}

assert_learning_rate() {
  local run_cfg="$1"
  local learning_rate="$2"

  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
values = re.findall(
    r"(?m)^  learning_rate:\s*(\S+)\s*$",
    path.read_text(encoding="utf-8"),
)

if len(values) != 1:
    raise SystemExit(
        f"Expected exactly one training.learning_rate entry, found {len(values)}"
    )

if Decimal(values[0]) != expected:
    raise SystemExit(
        f"training.learning_rate mismatch: expected={expected}, actual={values[0]}"
    )

print(f"Confirmed training.learning_rate={values[0]} in {path}")
' "$run_cfg" "$learning_rate"
}

run_one_experiment() {
  local source_cfg="$1"
  local learning_rate="$2"
  local cfg_name
  local normalized_learning_rate
  local lock_key
  local lock_file
  local run_cfg
  local train_status
  local discover_status
  local evaluate_status

  cfg_name="$(basename "$source_cfg" .yaml)"
  normalized_learning_rate="$(normalize_learning_rate "$learning_rate")"
  lock_key="${cfg_name}_${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already running config=$source_cfg, learning_rate=$learning_rate" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-${lock_key}.XXXXXX.yaml")"

  cleanup_run_cfg() {
    rm -f -- "$run_cfg"
  }

  if ! cp -- "$source_cfg" "$run_cfg"; then
    echo "[FAILED] Failed to create temporary config for $source_cfg" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  if ! override_learning_rate "$run_cfg" "$learning_rate"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  if ! assert_learning_rate "$run_cfg" "$learning_rate"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[TRAIN] config=$source_cfg learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"

  set +e
  torchrun \
    --standalone \
    --nproc_per_node="$NGPU" \
    scripts/train_ann.py \
    --config "$run_cfg"
  train_status=$?
  set -e

  if (( train_status != 0 )); then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  echo "[DISCOVER_PREFIX] config=$source_cfg learning_rate=$learning_rate"
  set +e
  python scripts/discover_prefix.py \
    --config "$run_cfg" \
    --stage post_finetuning
  discover_status=$?
  set -e

  if (( discover_status != 0 )); then
    echo "[FAILED] Prefix discovery failed; skipping ANN evaluation for config=$source_cfg, learning_rate=$learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$discover_status"
  fi

  echo "[WAIT] Waiting 10 seconds before ANN evaluation."
  sleep 10

  echo "[EVALUATE] config=$source_cfg learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$run_cfg" \
    --neuron ann
  evaluate_status=$?
  set -e

  cleanup_run_cfg
  flock -u 9
  exec 9>&-

  return "$evaluate_status"
}

declare -a FAILED_EXPERIMENTS=()
declare -a SUCCESSFUL_EXPERIMENTS=()
TOTAL_RUNS=$(( ${#ALL_CFGS[@]} * ${#LEARNING_RATES[@]} ))
CURRENT_RUN=0

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Learning rates       : ${LEARNING_RATES[*]}"
echo "Configs              : ${ALL_CFGS[*]}"
echo "Total runs           : $TOTAL_RUNS"

for cfg in "${ALL_CFGS[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    experiment="config=$cfg, learning_rate=$learning_rate"

    echo
    echo "================================================================"
    echo "Experiment $CURRENT_RUN / $TOTAL_RUNS: $experiment"
    echo "================================================================"

    if run_one_experiment "$cfg" "$learning_rate"; then
      SUCCESSFUL_EXPERIMENTS+=("$experiment")
    else
      FAILED_EXPERIMENTS+=("$experiment")
      echo "[WARNING] Failed: $experiment. Continuing to the next experiment." >&2
    fi
  done
done

echo
echo "Completed $TOTAL_RUNS experiments: ${#SUCCESSFUL_EXPERIMENTS[@]} succeeded, ${#FAILED_EXPERIMENTS[@]} failed."

if (( ${#FAILED_EXPERIMENTS[@]} > 0 )); then
  printf 'Failed experiments:\n'
  printf '  %s\n' "${FAILED_EXPERIMENTS[@]}"
  exit 1
fi
