#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_adult_lr.sh — Run SVM-X explanation pipeline
# Dataset : UCI Adult (Census Income)
# Target  : Logistic Regression classifier
#
# Expected interface of run_local_explanations.py:
#   --dataset   : "adult" or "bank" (uses synthetic demo data if CSV unavailable)
#   --model     : "rf", "xgb", "lr", "dt"
#   --n_samples : number of neighbourhood samples for SVM-X
#   --top_k     : number of top features to report
#   --seed      : random seed for reproducibility
#   --output_dir: directory for JSON results
#
# Output: a JSON file with fidelity metrics and top-K feature weights.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Environment check ────────────────────────────────────────────────
echo "=== Environment ==="
python --version
pip show scikit-learn 2>/dev/null | grep Version || echo "scikit-learn: not found"
echo "==================="

# ── Configuration ────────────────────────────────────────────────────
DATASET="adult"
MODEL="lr"
N_SAMPLES=2000
TOP_K=5
SEED=42
OUTPUT_DIR="outputs/adult"

mkdir -p "$OUTPUT_DIR"

# ── Run experiment ───────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting SVM-X explanation: dataset=${DATASET}, model=${MODEL}"

python -m src.svmx.experiments.run_local_explanations \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --n_samples "$N_SAMPLES" \
    --top_k "$TOP_K" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

echo "[$(date +%H:%M:%S)] Results saved to ${OUTPUT_DIR}/"
echo "Done."