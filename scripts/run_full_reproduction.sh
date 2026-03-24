#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_full_reproduction.sh — Run all implemented experiment configurations
#
# NOTE: This is a minimal multi-run scaffold, NOT a complete benchmark
# suite.  It sequentially launches the two currently configured
# dataset/model combinations.  A full reproduction of the paper's
# results would require additional models (XGB, DT, DNN), datasets
# (Amazon), and comparison baselines (LIME, LEMNA).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " SVM-X Reproduction — Minimal Multi-Run"
echo " $(date)"
echo "============================================"
echo ""

# ── Run 1: Bank Marketing + Random Forest ────────────────────────────
echo ">>> [1/2] Bank + RF"
bash "${SCRIPT_DIR}/run_bank_rf.sh"
echo ""

# ── Run 2: Adult + Logistic Regression ───────────────────────────────
echo ">>> [2/2] Adult + LR"
bash "${SCRIPT_DIR}/run_adult_lr.sh"
echo ""

echo "============================================"
echo " All runs complete.  Results in outputs/"
echo "============================================"