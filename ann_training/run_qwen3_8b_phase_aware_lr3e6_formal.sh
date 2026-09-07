#!/usr/bin/env bash
# Formal lr=3e-6 trial against the completed lr=2e-6 Phase control.
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/run_qwen3_8b_phase_aware_lr3e6_formal.sh
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
# One new formal candidate; previous formal Phase control is evaluation-only.
MATRIX="formal_lr3e6 4 4 3e-6 0.2 true"
exec "$SNN2_PYTHON" scripts/phase_aware_sweep.py \
  --matrix-text "$MATRIX" --audit-gradients --require-clip-true \
  --train-samples 10000 --test-samples 1000 \
  --phase-reference-config "${PHASE_CONTROL_CFG:-artifacts/phase_aware_round2_formal_cliptrue_v3/configs/p01_g4_s4.0_lr2e-06_w0.2_clip1.yaml}" \
  --source-config "${SOURCE_CFG:-configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml}" \
  --output "${SWEEP_DIR:-artifacts/phase_aware_lr3e6_formal_v1}" "$@"
