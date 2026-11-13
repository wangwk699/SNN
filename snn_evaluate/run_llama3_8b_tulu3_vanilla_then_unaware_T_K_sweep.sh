#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wangwenkang/SNN"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Starting Llama-3-8B Tulu-3 vanilla T/K sweep on GPUs 1,2,3"
echo "============================================================"
CUDA_VISIBLE_DEVICES=1,2,3 ./snn_evaluate/evaluate_llama3_8b_tulu3_vanilla_T_K_sweep.sh

echo "============================================================"
echo "Vanilla sweep finished; starting unaware T/K sweep on GPUs 1,2,3"
echo "============================================================"
CUDA_VISIBLE_DEVICES=1,2,3 ./snn_evaluate/evaluate_llama3_8b_tulu3_unaware_T_K_sweep.sh

echo "All Llama-3-8B Tulu-3 T/K sweeps completed."
