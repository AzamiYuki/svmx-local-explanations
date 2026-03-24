# Model Card: SVM-X Local Explainer

## Overview

SVM-X is a model-agnostic local explanation method that fits a linear Support Vector Machine as a surrogate in the neighbourhood of a target instance, then extracts the SVM's weight vector as a feature-importance explanation. This implementation follows Algorithm 1 from Xu et al. (Symmetry 2025).

## Intended Use

SVM-X is intended for **post-hoc local explanation** of binary classification models on tabular data. Given a trained black-box model and a single instance, it produces a ranked list of feature importances indicating which features most influenced the black-box prediction for that instance.

It is designed for research and analysis purposes — not for autonomous decision-making in safety-critical settings.

## Inputs

- **Target record**: a single feature vector (1-D numpy array) to explain.
- **Black-box model**: any classifier exposing `predict()` and `predict_proba()` (sklearn-compatible API).
- **Feature stats**: metadata describing feature ranges and one-hot group structure, produced by the preprocessing pipeline.

## Outputs

A dictionary containing:

- `top_k_indices` / `top_k_weights`: the K most important features and their normalised weights.
- `all_weights`: full L1-normalised weight vector from the linear SVM surrogate.
- `neighbours`: the generated neighbourhood samples (useful for fidelity evaluation).
- `surrogate`: the fitted sklearn `SVC` object.

## Method Summary

1. **Neighbourhood generation**: perturb the target record by randomly replacing feature values. One-hot groups are sampled as valid categories (exactly one active column per group). Continuous features are sampled uniformly within their observed range.
2. **Distance-based weighting**: each neighbour receives a weight based on how similarly the black-box treats it compared to the target record, using the prediction-probability distance from Eq. 8 and exponential decay from Eq. 7 in the paper.
3. **Surrogate fitting**: a linear SVM (`sklearn.svm.SVC` with `kernel="linear"`) is trained on the neighbourhood, weighted by proximity.
4. **Weight extraction**: the SVM's coefficient vector is L1-normalised and ranked by absolute value.

## Limitations

- **Linear surrogate**: the linear kernel limits the surrogate's own capacity. Highly nonlinear local boundaries may be underfitted.
- **Loss mismatch**: the paper formulates training as weighted MSE minimisation (Eq. 9), while sklearn's SVC uses hinge loss. This is a pragmatic approximation, not an exact reproduction.
- **Degenerate regions**: if the black-box is very confident near the target, all neighbours may receive the same label. The code applies a small label-flip heuristic, but the resulting explanation carries less signal.
- **One-hot group inference**: relies on feature name prefixes from `pd.get_dummies`. Unusual naming patterns could cause misclassification of groups, though the implementation only groups columns already marked as categorical.

## Implementation Notes

- Built with scikit-learn 1.3+; no GPU required.
- Typical runtime: 10–30 seconds per target instance depending on dimensionality and `n_samples`.
- This is a reproduction-oriented implementation, not official code from the paper authors.

## Reference

> Xu, J. et al. "Towards Faithful Local Explanations: Leveraging SVM to Interpret Black-Box Machine Learning Models." *Symmetry*, 17(6), 950, 2025.