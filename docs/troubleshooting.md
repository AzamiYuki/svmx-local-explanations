# Troubleshooting

## 1. `xgboost` is not installed
If `xgboost` is unavailable, the repository may fall back to a simpler model depending on the current implementation. For consistent experiments, install the package listed in `requirements.txt`.

## 2. Train/test encoded columns do not match
If one-hot encoded columns differ between splits, make sure preprocessing reuses the training column layout through the `expected_columns` mechanism in `preprocess.py`.

## 3. SVM surrogate fails because all local labels are the same
In highly stable local regions, all neighbourhood samples may receive the same black-box label. This makes local boundary fitting difficult. Reduce neighbourhood size, perturb more aggressively, or choose a different target instance.

## 4. Invalid categorical perturbations
If local samples look unrealistic, check that one-hot groups are being perturbed as grouped categorical features rather than as independent binary columns.

## 5. Fidelity metrics look unusually high or low
Confirm that:
- the same neighbourhood samples are used for both black-box and surrogate evaluation,
- the target model is fitted before explanation,
- probability outputs are aligned by class order.