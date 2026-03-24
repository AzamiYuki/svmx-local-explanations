# SVM-X: Local Explanations via SVM Surrogates

> **Reproduction-oriented implementation** — this is **not** official code from the paper authors.  
> Based on: *"Towards Faithful Local Explanations: Leveraging SVM to Interpret Black-Box Machine Learning Models"* (Xu et al., Symmetry 2025).

## Overview

SVM-X is a model-agnostic local explanation method that uses a linear Support Vector Machine as a surrogate to approximate a black-box model's decision boundary in the neighbourhood of a target instance. The original paper suggests that the margin-maximising property of SVMs can yield explanations with improved local fidelity and weight stability compared to linear surrogates (e.g. LIME) or mixture-based approaches (e.g. LEMNA).

This repository implements a **minimal, research-oriented version of the SVM-X pipeline** for tabular datasets (UCI Adult, Bank Marketing), including preprocessing, neighbourhood sampling, surrogate fitting, feature-weight extraction, and local fidelity evaluation.

The codebase is intentionally kept lightweight and modular, and is structured so that additional datasets or explanation baselines could be integrated in the future. The current implementation focuses on demonstrating the core SVM-X workflow rather than providing a full benchmarking framework.

**What is implemented:**
- SVM-X algorithm (based on Algorithm 1 from the paper)
- Prediction-probability-based distance metric (Eq. 8) and exponential sample weighting (Eq. 7)
- Group-aware one-hot perturbation (ensuring valid categorical samples)
- Local fidelity evaluation (accuracy, F1, MSE) and weight stability metric
- Minimal experiment runner with JSON output

**What is not yet implemented:**
- Explicit baseline pipelines (e.g. LIME, LEMNA)
- Text-based datasets (e.g. Amazon Reviews)
- Neural network (DNN) target models

## Project Structure


├── src/svmx/
│ ├── data/
│ │ └── preprocess.py # Normalisation, encoding, missing values
│ ├── explainers/
│ │ ├── svmx.py # Core SVM-X explainer
│ │ └── local_sampling.py # Neighbourhood perturbation & weighting
│ ├── evaluation/
│ │ └── fidelity.py # Accuracy, F1, MSE, stability metrics
│ └── experiments/
│ └── run_local_explanations.py # Minimal experiment entry point
├── scripts/
│ └── run_bank_rf.sh # Reproducible experiment launcher
├── tests/
│ └── test_preprocess.py # Unit tests for preprocessing
├── docs/
│ └── questionnaire_notes.md # Structured notes for M2 application
├── requirements.txt
└── README.md


## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

Minimal requirements.txt:

numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
pytest>=7.4
Quick Start
# Run a minimal SVM-X experiment on the Bank dataset with a Random Forest target
bash scripts/run_bank_rf.sh

# Or run directly (uses synthetic demo data when real CSVs are unavailable):
python -m src.svmx.experiments.run_local_explanations \
    --dataset bank --model rf --n_samples 2000 --top_k 5 --seed 42

The experiment script trains a black-box model, explains a single target instance with SVM-X, evaluates local fidelity, and saves a JSON results file.

Evaluation Metrics
Metric	What it measures
Accuracy	Fraction of local samples where surrogate and black-box agree on class
F1 Score	Harmonic mean of precision and recall on surrogate vs. black-box labels
MSE	Mean squared error between surrogate and black-box predicted probabilities
Weight variance	Mean variance of feature weights across nearby target records (stability)

These metrics follow Section 4.1.3 of the paper. The target model's predictions are treated as the reference for evaluating the surrogate locally.

My Contribution

I built this repository from scratch as a personal research project to:

Implement the SVM-X algorithm based on the paper's mathematical description, including the prediction-probability distance metric and exponential weighting scheme.

Design a modular and reproducible pipeline for tabular data, covering preprocessing, sampling, surrogate fitting, and evaluation.

Address underspecified details in the paper (e.g. handling one-hot features during perturbation, dealing with degenerate local regions) with explicit and documented implementation choices.

Highlight practical differences between the paper's formulation (weighted MSE objective) and the sklearn-based approximation (hinge-loss SVM).

All code, experiment scripts, and documentation are my own work. The paper's authors did not release official code; this implementation is based solely on the published methodology.

License

MIT — for educational and research purposes.