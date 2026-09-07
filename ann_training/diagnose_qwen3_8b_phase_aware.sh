#!/usr/bin/env bash
# Read-only model diagnostics: no training/optimizer steps.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SNN2_PYTHON="${SNN2_PYTHON:-/home/wangwenkang/miniconda3/envs/snn2/bin/python}"
export PATH="$(dirname -- "$SNN2_PYTHON"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
exec "$SNN2_PYTHON" scripts/diagnose_phase_aware.py \
  --formal-dir "${FORMAL_DIR:-artifacts/phase_aware_formal_v1}" \
  --output "${DIAGNOSTIC_DIR:-artifacts/phase_diagnostics_v1}/calibration_group_size_4" \
  --samples 128 --smoke-samples 16 --slopes 0.5 1 2 4 8 --top-k 3 "$@"
