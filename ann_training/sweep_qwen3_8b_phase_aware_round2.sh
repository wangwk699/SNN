#!/usr/bin/env bash
# Controlled second-round Phase-aware sweep.
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/sweep_qwen3_8b_phase_aware_round2.sh
# Add --dry-run to preview; add --summarize-only to rebuild reports.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SNN2_PYTHON="${SNN2_PYTHON:-/home/wangwenkang/miniconda3/envs/snn2/bin/python}"
if [[ ! -x "$SNN2_PYTHON" ]]; then
  echo "Set SNN2_PYTHON to the snn2 environment's Python executable." >&2
  exit 1
fi
export PATH="$(dirname -- "$SNN2_PYTHON"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# Fixed T=4/4, K=6, G4, seed42, max_grad_norm=1; 14 controlled trials.
MATRIX="slope_warmup 4 3 1e-6 0.1 true
slope_warmup 4 3 1e-6 0.2 true
slope_warmup 4 4 1e-6 0.1 true
slope_warmup 4 4 1e-6 0.2 true
slope_warmup 4 5 1e-6 0.1 true
slope_warmup 4 5 1e-6 0.2 true
slope_warmup 4 6 1e-6 0.1 true
slope_warmup 4 6 1e-6 0.2 true
learning_rate 4 4 5e-7 0.2 true
learning_rate 4 4 2e-6 0.2 true
learning_rate 4 5 5e-7 0.2 true
learning_rate 4 5 2e-6 0.2 true
clip_control 4 4 1e-6 0.2 false
clip_control 4 5 1e-6 0.2 false"
exec "$SNN2_PYTHON" scripts/phase_aware_sweep.py \
  --matrix-text "$MATRIX" --audit-gradients \
  --train-samples 1024 --test-samples 128 \
  --source-config "${SOURCE_CFG:-configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml}" \
  --output "${SWEEP_DIR:-artifacts/phase_aware_round2_v1}" "$@"
