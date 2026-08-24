#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
SOURCE_CFG="configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml"

# ---------------------------------------------------------------------------
# 学习率列表：脚本内部会依次遍历这些取值
# ---------------------------------------------------------------------------
LEARNING_RATES=(1.0e-07 1.0e-06 1.0e-05 2.0e-05 5.0e-05 8.0e-05 1.0e-04)

# GPU 列表通过环境变量 CUDA_VISIBLE_DEVICES 指定，默认 "2,3"
gpu_devices="${CUDA_VISIBLE_DEVICES:-2,3}"

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

# ---------------------------------------------------------------------------
# 辅助函数：覆盖配置中的学习率
# ---------------------------------------------------------------------------
override_learning_rate() {
  local cfg_path="$1"
  local lr="$2"
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
' "$cfg_path" "$lr"
}

# ---------------------------------------------------------------------------
# 辅助函数：验证配置中的学习率是否与期望一致
# ---------------------------------------------------------------------------
assert_learning_rate() {
  local cfg_path="$1"
  local lr="$2"
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
' "$cfg_path" "$lr"
}

# ---------------------------------------------------------------------------
# 主执行函数：处理单个学习率的完整流程
# 参数：$1 = 学习率值
# 该函数会在子 shell 中运行，因此可以使用 return 1 来表示失败，
# 子 shell 的退出状态会被外层循环捕获。
# ---------------------------------------------------------------------------
run_one_learning_rate() {
  local learning_rate="$1"

  # 规范化学习率用于锁文件和临时文件名
  local normalized_learning_rate
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

  local lock_key="${normalized_learning_rate//[^0-9A-Za-z_.-]/_}"
  local LOCK_FILE="/tmp/snn-qwen3-1.7b-phase-aware-lr-${lock_key}.lock"

  # 尝试获取锁，如果失败说明已有同学习率任务在运行，直接跳过
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another instance is already using learning_rate=$learning_rate" >&2
    return 1
  fi

  # 创建临时配置文件
  local RUN_CFG
  RUN_CFG="$(mktemp "/tmp/snn-qwen3-1.7b-phase-aware-${lock_key}.XXXXXX.yaml")"
  cp -- "$SOURCE_CFG" "$RUN_CFG"

  # 设置返回时的清理 trap（仅在函数内有效）
  trap 'rm -f -- "$RUN_CFG"' RETURN

  echo "Starting PID=$$ learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU"
  echo "Per-instance config: $RUN_CFG"

  # 覆盖并验证学习率
  override_learning_rate "$RUN_CFG" "$learning_rate"
  assert_learning_rate "$RUN_CFG" "$learning_rate"

  # 训练
  torchrun \
    --standalone \
    --nproc_per_node="$NGPU" \
    scripts/train_ann.py \
    --config "$RUN_CFG"

  # 训练结束后再次覆盖并验证，确保评估配置一致
  override_learning_rate "$RUN_CFG" "$learning_rate"
  assert_learning_rate "$RUN_CFG" "$learning_rate"

  # 评估
  accelerate launch --num_processes "$NGPU" \
    scripts/evaluate_tldr.py \
    --config "$RUN_CFG" \
    --neuron ann

  echo "Completed PID=$$ learning_rate=$learning_rate GPUs=$CUDA_VISIBLE_DEVICES"
}

# ---------------------------------------------------------------------------
# 主循环：依次遍历所有学习率，单个失败不影响后续
# ---------------------------------------------------------------------------
echo "Starting learning rate sweep: ${LEARNING_RATES[*]}"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES (NGPU=$NGPU)"

for lr in "${LEARNING_RATES[@]}"; do
  echo "=============================================================="
  echo "Processing learning_rate=$lr"
  echo "=============================================================="

  # 在子 shell 中运行，即使内部失败也只影响子 shell，
  # 外层用 if 捕获状态并继续下一个学习率。
  if ! (run_one_learning_rate "$lr"); then
    echo "!!! Learning rate $lr failed or was skipped, continuing to next !!!" >&2
  fi
done

echo "All learning rates processed."