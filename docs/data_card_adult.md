# Data Card: UCI Adult (Census Income)

## Dataset Summary

The UCI Adult dataset (also known as Census Income) contains demographic information extracted from the 1994 U.S. Census Bureau database. It is a standard benchmark for binary classification on tabular data.

- **Source**: UCI Machine Learning Repository
- **Instances**: 48,842 (32,561 train / 16,281 test in the original split)
- **Features**: 14 attributes (6 continuous, 8 categorical)
- **Task**: predict whether an individual's annual income exceeds $50K

## Prediction Target

- Column: `income`
- Classes: `<=50K` (majority, ~76%) and `>50K` (~24%)
- Encoded as: 0 / 1 in this repository

## Feature Types

**Continuous** (6): `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

**Categorical** (8): `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

## Preprocessing in This Repository

Handled by `src/svmx/data/preprocess.py` with `dataset_name="adult"`:

1. **Missing values**: entries marked `?` are replaced with the column mode (categorical) or median (continuous).
2. **One-hot encoding**: all 8 categorical columns are expanded via `pd.get_dummies`, producing approximately 100+ binary indicator columns depending on the value set present.
3. **Min-max normalisation**: the 6 continuous columns are scaled to [0, 1] using `sklearn.preprocessing.MinMaxScaler`.
4. **Column alignment**: an `expected_columns` parameter allows the test set to be aligned to training columns, preventing one-hot mismatch.

## Known Limitations and Bias Concerns

- The dataset reflects 1994 U.S. census demographics and contains known historical biases related to race, sex, and national origin. Models trained on this data may encode or amplify these biases.
- The `fnlwgt` column (final sampling weight) is a census-specific construct and is not a meaningful individual-level feature; it is retained for consistency with the paper's experimental setup.
- The class distribution is imbalanced (~76% negative), which can affect evaluation metrics. This repository uses accuracy, F1, and recall to provide a balanced view.
- This repository uses the Adult dataset strictly for evaluating explanation fidelity, not for making real-world income predictions.