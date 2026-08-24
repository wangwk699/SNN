#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml"

# Per-instance defaults. Override them through environment variables when launching.
learning_rate="${LEARNING_RATE:-1.0e-07}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-2,3}"

cd "$PROJECT_ROOT"

if [[ ! -f "$SOURCE_CFG" ]]; then
  echo "Missing generated config: $PROJECT_ROOT/$SOURCE_CFG" >&2
  exit 1
fi

if [[ -z "$learning_rate" || -z "$gpu_devices" ]]; then
  echo "LEARNING_RATE and CUDA_VISIBLE_DEVICES must be non-empty" >&2
  exit 1
fi

IFS=',' read -r -a gpu_list <<< "$gpu_devices"
NGPU="${#gpu_list[@]}"
if ((NGPU <= 0)); then
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

# Different learning rates map to different artifact runs. Prevent accidental
# concurrent writers for the same learning rate while allowing different rates.
normalized_learning_rate="$(python3 -c '
from decimal import Decimal, InvalidOperation
import sys
try:
    value = Decimal(sys.argv[1])
except InvalidOperation as exc:
    raise SystemExit(f"Invalid LEARNING_RATE: {sys.argv[1]}") from exc
if not value.is_finite() or value <= 0:
    raise SystemExit("LEARNING_RATE must be a positive finite number")
print(format(value.normalize(), "E"))
' "$learning_rate")"
lock_key="${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
LOCK_FILE="/tmp/snn-qwen3-1.7b-phase-aware-lr-${lock_key}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another instance is already using learning_rate=$learning_rate" >&2
  exit 1
fi

RUN_CFG="$(mktemp "/tmp/snn-qwen3-1.7b-phase-aware-${lock_key}.XXXXXX.yaml")"
cp -- "$SOURCE_CFG" "$RUN_CFG"

cleanup() {
  rm -f -- "$RUN_CFG"
}
trap cleanup EXIT

override_learning_rate() {
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
path.write_text(
    updated + ("" if updated.endswith("\n") else "\n"),
    encoding="utf-8",
)
' "$RUN_CFG" "$learning_rate"
}

assert_learning_rate() {
  python3 -c '
from decimal import Decimal
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected = Decimal(sys.argv[2])
text = path.read_text(encoding="utf-8")
values = re.findall(r"(?m)^  learning_rate:\s*(\S+)\s*$", text)
if len(values) != 1:
    raise SystemExit(
        f"Expected exactly one training.learning_rate entry, found {len(values)}"
    )
actual = Decimal(values[0])
if actual != expected:
    raise SystemExit(
        f"training.learning_rate mismatch: expected={expected}, actual={actual}"
    )
print(f"Confirmed training.learning_rate={values[0]} in {path}")
' "$RUN_CFG" "$learning_rate"
}

echo "Starting PID=$$ learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
echo "Per-instance config: $RUN_CFG"

override_learning_rate
assert_learning_rate

torchrun \
  --standalone \
  --nproc_per_node="$NGPU" \
  scripts/train_ann.py \
  --config "$RUN_CFG"

# Re-apply and verify the same per-instance override immediately before evaluation.
override_learning_rate
assert_learning_rate

accelerate launch --num_processes "$NGPU" \
  scripts/evaluate_tldr.py \
  --config "$RUN_CFG" \
  --neuron ann

echo "Completed PID=$$ learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES"
