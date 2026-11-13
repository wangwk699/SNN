#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml"

# Phase timesteps to run sequentially.
TS=(3 5 6)

# Learning rates to run sequentially.
LEARNING_RATES=(
  5.0e-06
)

# CUDA_VISIBLE_DEVICES is provided externally.
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/llama3_8b/train_and_evaluate_llama3_8b_tulu3_phase_aware_LEARNING_RATE_T.sh
gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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

set_learning_rate() {
  local run_cfg="$1"
  local learning_rate="$2"
  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
text = path.read_text(encoding="utf-8")
updated, count = re.subn(r"(?m)^(  learning_rate:)\s*\S+\s*$", rf"\1 {sys.argv[2]}", text)
if count != 1:
    raise SystemExit(f"Expected exactly one training.learning_rate entry, found {count}")
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
values = re.findall(r"(?m)^  learning_rate:\s*(\S+)\s*$", updated)
if len(values) != 1 or Decimal(values[0]) != expected:
    raise SystemExit(f"training.learning_rate mismatch: expected={expected}, actual={values}")
print(f"Confirmed training.learning_rate={values[0]} in {path}")
' "$run_cfg" "$learning_rate"
}

override_phase_T() {
  local run_cfg="$1"
  local phase_T="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
value = sys.argv[2]
if re.fullmatch(r"[1-9][0-9]*", value) is None:
    raise SystemExit("phase.T must be a positive integer")

text = path.read_text(encoding="utf-8")
sections = list(re.finditer(r"(?ms)^phase:[ \t]*\n.*?(?=^[^ \t\r\n#]|\Z)", text))
if len(sections) != 1:
    raise SystemExit(f"Expected exactly one phase section, found {len(sections)}")
section = sections[0]
updated, count = re.subn(
    r"(?m)^(  T:)[ \t]*\S+[^\n]*$",
    rf"\1 {value}",
    section.group(),
)
if count != 1:
    raise SystemExit(f"Expected exactly one phase.T entry, found {count}")
text = text[:section.start()] + updated + text[section.end():]
path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
' "$run_cfg" "$phase_T"
}

assert_phase_T() {
  local run_cfg="$1"
  local phase_T="$2"

  python3 -c '
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
sections = re.findall(r"(?ms)^phase:[ \t]*\n.*?(?=^[^ \t\r\n#]|\Z)", path.read_text(encoding="utf-8"))
if len(sections) != 1:
    raise SystemExit(f"Expected exactly one phase section, found {len(sections)}")
values = re.findall(r"(?m)^  T:[ \t]*([0-9]+)[ \t]*$", sections[0])
if len(values) != 1 or values[0] != expected:
    raise SystemExit(f"phase.T mismatch: expected={expected}, actual={values}")
print(f"Confirmed phase.T={values[0]} in {path}")
' "$run_cfg" "$phase_T"
}

configure_run() {
  local run_cfg="$1"
  local learning_rate="$2"
  local phase_T="$3"

  set_learning_rate "$run_cfg" "$learning_rate" \
    && override_phase_T "$run_cfg" "$phase_T" \
    && assert_phase_T "$run_cfg" "$phase_T"
}

run_one_learning_rate() {
  local learning_rate="$1" phase_T="$2" normalized_learning_rate lock_key lock_file run_cfg train_status eval_status
  if [[ ! "$phase_T" =~ ^[1-9][0-9]*$ ]]; then
    echo "[FAILED] Invalid phase.T: $phase_T" >&2
    return 1
  fi
  if ! normalized_learning_rate="$(normalize_learning_rate "$learning_rate")"; then
    return 1
  fi
  lock_key="T${phase_T}-${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
  lock_file="/tmp/snn-llama3-8b-tulu3-phase-aware-lr-${lock_key}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[FAILED] Another instance is already using phase.T=$phase_T learning_rate=$learning_rate" >&2
    exec 9>&-
    return 1
  fi

  run_cfg="$(mktemp "/tmp/snn-llama3-8b-tulu3-phase-aware-${lock_key}.XXXXXX.yaml")"
  cleanup_run_cfg() { rm -f -- "$run_cfg"; }

  if ! cp -- "$SOURCE_CFG" "$run_cfg" || ! configure_run "$run_cfg" "$learning_rate" "$phase_T"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  # Match the phase-aware Qwen3 template's allocator setting.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "[TRAIN] phase.T=$phase_T learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  torchrun --standalone --nproc_per_node="$NGPU" scripts/train_ann.py --config "$run_cfg"
  train_status=$?
  set -e
  if (( train_status != 0 )); then
    echo "[FAILED] ANN training failed; skipping lm-eval for phase.T=$phase_T learning_rate=$learning_rate" >&2
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return "$train_status"
  fi

  # Re-apply and verify learning_rate and phase.T before evaluation.
  if ! configure_run "$run_cfg" "$learning_rate" "$phase_T"; then
    cleanup_run_cfg
    flock -u 9
    exec 9>&-
    return 1
  fi

  echo "[EVAL] phase.T=$phase_T learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  set +e
  accelerate launch --num_processes "$NGPU" scripts/evaluate_lm_harness.py \
    --config "$run_cfg" --neuron ann
  eval_status=$?
  set -e

  cleanup_run_cfg
  flock -u 9
  exec 9>&-
  return "$eval_status"
}

declare -a FAILED_COMBINATIONS=()
declare -a SUCCESSFUL_COMBINATIONS=()
TOTAL_RUNS="$((${#TS[@]} * ${#LEARNING_RATES[@]}))"
CURRENT_RUN=0
echo "Llama-3-8B Tulu-3 Phase-Aware ANN Training + lm-eval"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "NGPU                 : $NGPU"
echo "Phase T values       : ${TS[*]}"
echo "Learning rates       : ${LEARNING_RATES[*]}"
echo "Total combinations   : $TOTAL_RUNS"

for phase_T in "${TS[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    combination="phase.T=$phase_T,learning_rate=$learning_rate"
    echo "Experiment $CURRENT_RUN / $TOTAL_RUNS: $combination"
    if run_one_learning_rate "$learning_rate" "$phase_T"; then
      SUCCESSFUL_COMBINATIONS+=("$combination")
    else
      FAILED_COMBINATIONS+=("$combination")
      echo "[WARNING] $combination failed; continuing." >&2
    fi
  done
done

echo "Completed $TOTAL_RUNS configuration combinations: ${#SUCCESSFUL_COMBINATIONS[@]} succeeded, ${#FAILED_COMBINATIONS[@]} failed."
if (( ${#FAILED_COMBINATIONS[@]} > 0 )); then
  printf 'Failed combinations:\n  %s\n' "${FAILED_COMBINATIONS[@]}"
  exit 1
fi
