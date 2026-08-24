#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
CFG_8_U="configs/generated/exp1_qwen3_8b_tldr__unaware.yaml"
LEARNING_RATES=(1.0e-08 1.0e-07 1.0e-06 1.0e-05)
NGPU=4

cd "$PROJECT_ROOT"

if [[ ! -f "$CFG_8_U" ]]; then
  echo "Missing generated config: $PROJECT_ROOT/$CFG_8_U" >&2
  exit 1
fi

CONFIG_BACKUP="$(mktemp /tmp/snn-qwen3-8b-unaware-config.XXXXXX.yaml)"
cp -- "$CFG_8_U" "$CONFIG_BACKUP"

restore_config() {
  cp -- "$CONFIG_BACKUP" "$CFG_8_U"
  rm -f -- "$CONFIG_BACKUP"
}
trap restore_config EXIT

export CUDA_VISIBLE_DEVICES=4,5,6,7

failure_count=0

record_failure() {
  local learning_rate="$1"
  local stage="$2"
  local status="$3"
  failure_count=$((failure_count + 1))
  echo "[$(date --iso-8601=seconds)] FAILED learning_rate=$learning_rate stage=$stage exit_status=$status; continuing" >&2
}

for learning_rate in "${LEARNING_RATES[@]}"; do
  echo "[$(date --iso-8601=seconds)] Starting learning_rate=$learning_rate"

  conda run --no-capture-output -n snn2 python -c '
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
    raise SystemExit(f"Expected exactly one training.learning_rate entry, found {count}")
path.write_text(updated + ("" if updated.endswith("\n") else "\n"), encoding="utf-8")
' "$CFG_8_U" "$learning_rate"
  update_status=$?
  if ((update_status != 0)); then
    record_failure "$learning_rate" "update_config" "$update_status"
    continue
  fi

  conda run --no-capture-output -n snn2 torchrun \
    --standalone --nproc_per_node="$NGPU" \
    scripts/train_ann.py --config "$CFG_8_U"
  train_status=$?
  if ((train_status != 0)); then
    record_failure "$learning_rate" "train_ann" "$train_status"
  fi

  conda run --no-capture-output -n snn2 python scripts/discover_prefix.py \
    --config "$CFG_8_U" --stage post_finetuning
  prefix_status=$?
  if ((prefix_status != 0)); then
    record_failure "$learning_rate" "discover_prefix" "$prefix_status"
  fi

  conda run --no-capture-output -n snn2 accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py --config "$CFG_8_U" --neuron ann
  evaluation_status=$?
  if ((evaluation_status != 0)); then
    record_failure "$learning_rate" "evaluate_tldr" "$evaluation_status"
  fi

  echo "[$(date --iso-8601=seconds)] Completed learning_rate=$learning_rate"
done

if ((failure_count > 0)); then
  echo "Sweep completed with $failure_count failed stage(s); all scheduled stages were attempted." >&2
  exit 1
fi

echo "Sweep completed successfully."
