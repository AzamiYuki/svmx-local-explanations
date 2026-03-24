# Questionnaire Notes — SVM-X Reproduction Project

Structured answers for the M2 AI (Université Paris-Saclay) application form and oral interview.

---

## Repository URL

`https://github.com/<username>/svmx-local-explanations`

---

## My Role

I am the sole author of this repository. I designed the project structure, implemented the SVM-X algorithm from the paper's mathematical descriptions, wrote the evaluation pipeline, and documented the work. The paper's authors did not release official code; this implementation is based entirely on the published methodology.

---

## Paper Reference

> Xu, J. et al. "Towards Faithful Local Explanations: Leveraging SVM to Interpret Black-Box Machine Learning Models." *Symmetry*, 17(6), 950, 2025.

---

## Datasets

| Dataset | Source | Samples | Features | Task |
|---|---|---|---|---|
| UCI Adult | UCI ML Repository | 48 842 | 14 (mixed) | Income > $50K (binary) |
| Bank Marketing | UCI ML Repository | 45 211 | 17 (mixed) | Term deposit subscription (binary) |

I chose these two datasets because the paper reports its largest fidelity improvements on tabular data with mixed categorical and continuous columns, and because their structure exercises the preprocessing and one-hot group sampling logic thoroughly.

The paper also evaluates on the Amazon Product Review dataset (text); I have not yet integrated this dataset, but the repository structure would support extension to binary bag-of-words features.

---

## Reproducibility Strategy

1. **Fixed random seeds** (`--seed 42`) at every stochastic step: neighbourhood sampling, train/test split, SVM initialisation.
2. **Deterministic preprocessing**: imputation, encoding, and scaling are pure NumPy/pandas operations with no hidden randomness.
3. **Shell scripts** (`scripts/run_bank_rf.sh`) capture the full command line and print environment versions.
4. **Unit tests** (`tests/test_preprocess.py`) verify that the preprocessing pipeline is deterministic and that outputs are invariant across runs with the same seed.
5. **Train/test column alignment**: the `expected_columns` parameter in `preprocess()` prevents one-hot mismatch between splits.

---

## Model Choice

The repository is structured to evaluate SVM-X on Random Forest, XGBoost, Logistic Regression, and Decision Tree targets, all trained via scikit-learn. I focused initial small-scale runs on **Random Forest** because:

- The paper reports large fidelity gains for SVM-X on tree-based models, particularly RF and XGBoost.
- RF's structured nonlinearity is a natural fit for testing whether SVM-X's margin-based surrogate captures the local boundary better than a simple linear approximation.
- DNN targets are noted as a known weak spot in the paper itself; I have not implemented DNN experiments.

I have not yet run systematic comparisons against LIME. The evaluation pipeline is designed to support such comparisons once a LIME baseline module is added.

---

## Failure Cases & Limitations

1. **DNN targets**: The paper reports that SVM-X's advantage over LIME shrinks to ~4% for DNN targets, likely due to stochastic training dynamics. I have not run DNN experiments to verify this independently.
2. **High-cardinality categoricals**: One-hot encoding on the Adult dataset produces ~100+ columns. The linear SVM surrogate may overfit to noisy binary indicators when the neighbourhood sample count is low relative to dimensionality.
3. **Degenerate neighbourhoods**: When the black-box is very confident near x_t, all neighbours may receive the same label. The code handles this with a small label-flip heuristic, but the resulting explanation carries less signal.
4. **Surrogate loss mismatch**: The paper formulates surrogate training as weighted MSE minimisation (Eq. 9), while sklearn's SVC optimises hinge loss with sample weights. The two objectives are not identical; this is a pragmatic approximation documented in the code.
5. **One-hot group inference**: Group-aware sampling relies on feature name prefixes from `pd.get_dummies`. If a continuous column name happens to match a categorical prefix pattern, it could be misclassified. The current implementation mitigates this by only grouping columns already marked as categorical.

---

## Hardware

- CPU: Intel i7 / Apple M-series (no GPU required)
- RAM: 16 GB
- Typical runtime observed in small-scale runs: preprocessing + training a Random Forest + generating 2 000 neighbourhood samples + fitting the SVM surrogate takes on the order of 10–30 seconds per target instance, depending on dataset dimensionality.

---

## Evaluation Metrics

| Metric | Definition | Purpose |
|---|---|---|
| **Accuracy** | % of local samples where surrogate class = black-box class | Overall agreement in the neighbourhood |
| **Recall** | TP / (TP + FN) w.r.t. black-box labels | Whether the surrogate recovers the target class |
| **F1 Score** | Harmonic mean of precision and recall | Balanced single-number fidelity summary |
| **MSE** | Mean (p_blackbox − p_surrogate)² | Probability-level fidelity (finer-grained than class agreement) |
| **Weight variance** | Mean per-feature variance across nearby explanations (Eq. 11) | Explanation stability — lower is more consistent |

---

## Key Takeaway for Interview

SVM-X's central insight is that a **margin-maximising surrogate** (SVM) naturally prioritises the decision boundary region, whereas a least-squares surrogate (LIME's ridge regression) treats all local points equally. I implemented this repository to investigate whether that theoretical advantage translates into measurable fidelity and stability gains on real tabular data, and to develop a clean experimental setup that could support systematic comparison in future work.