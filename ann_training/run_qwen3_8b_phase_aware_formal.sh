#!/usr/bin/env bash
# Three formal candidates selected after the 50-trial quick sweep.
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/run_qwen3_8b_phase_aware_formal.sh
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
# family group_size surrogate_slope learning_rate warmup_ratio common_clip
# Priority 1: previous p05; priority 2: p19; priority 3: p47.
# Fixed phase.T=4, mtn.T=4, mtn.K=6, calibration.num_samples=128, seed=42.
MATRIX="formal_p05 4 1 1e-6 0.1 true
formal_p19 1 1 1e-6 0.1 false
formal_p47 4 4 1e-6 0.2 true"
exec "$SNN2_PYTHON" scripts/phase_aware_sweep.py \
  --matrix-text "$MATRIX" \
  --train-samples 10000 --test-samples 1000 \
  --source-config "${SOURCE_CFG:-configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml}" \
  --output "${SWEEP_DIR:-artifacts/phase_aware_formal_v1}" "$@"
