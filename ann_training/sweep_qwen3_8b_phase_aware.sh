#!/usr/bin/env bash
# 50 controlled quick trials, followed by analysis by the user/assistant.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/sweep_qwen3_8b_phase_aware.sh
# Preview: ... ./ann_training/sweep_qwen3_8b_phase_aware.sh --dry-run
# Summarize again: ... ./ann_training/sweep_qwen3_8b_phase_aware.sh --summarize-only
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Use snn2 even when launched from another activated conda environment.
SNN2_PYTHON="${SNN2_PYTHON:-/home/wangwenkang/miniconda3/envs/snn2/bin/python}"
if [[ ! -x "$SNN2_PYTHON" ]]; then
  echo "Set SNN2_PYTHON to the snn2 environment's Python executable." >&2
  exit 1
fi
export PATH="$(dirname -- "$SNN2_PYTHON"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# Every row is: family group_size slope learning_rate warmup_ratio common_clip.
# T=4, MTN T=4/K=6, 128 calibration samples, seed=42 are fixed by the helper.
# Core: 15; LR: 18; Clip: 6; warmup: 8; larger slopes: 2; G64 anchor: 1.
# No lr=1e-5: that setting already deteriorated severely in the previous trial.
MATRIX="$(
  for g in 1 4 16 32 128; do
    for slope in 0.5 1 4; do
      printf 'group_slope %s %s 1e-6 0.1 true\n' "$g" "$slope"
    done
  done
  for g in 1 4 16; do
    for slope in 1 4; do
      for lr in 2.5e-7 5e-7 2e-6; do
        printf 'learning_rate %s %s %s 0.1 true\n' "$g" "$slope" "$lr"
      done
      printf 'clip %s %s 1e-6 0.1 false\n' "$g" "$slope"
    done
  done
  for g in 1 4; do
    for slope in 1 4; do
      for warmup in 0.0 0.2; do
        printf 'warmup %s %s 1e-6 %s true\n' "$g" "$slope" "$warmup"
      done
    done
  done
  for slope in 8 16; do
    printf 'large_slope 1 %s 1e-6 0.1 true\n' "$slope"
  done
  printf 'group_anchor 64 1 1e-6 0.1 true\n'
)"
exec "$SNN2_PYTHON" scripts/phase_aware_sweep.py \
  --matrix-text "$MATRIX" \
  --source-config "${SOURCE_CFG:-configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml}" \
  --output "${SWEEP_DIR:-artifacts/phase_aware_sweep_v1}" "$@"
