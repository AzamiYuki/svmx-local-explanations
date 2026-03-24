# Data Card: Bank Marketing

## Dataset Summary

The Bank Marketing dataset contains information on direct marketing campaigns (phone calls) conducted by a Portuguese banking institution. It is a standard benchmark for binary classification on tabular data with mixed feature types.

- **Source**: UCI Machine Learning Repository (Moro et al., 2014)
- **Instances**: 45,211
- **Features**: 16 attributes (7 continuous, 9 categorical)
- **Task**: predict whether a client will subscribe to a term deposit

## Prediction Target

- Column: `y`
- Classes: `no` (majority, ~88%) and `yes` (~12%)
- Encoded as: 0 / 1 in this repository

## Feature Types

**Continuous** (7): `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`

**Categorical** (9): `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

## Preprocessing in This Repository

Handled by `src/svmx/data/preprocess.py` with `dataset_name="bank"`:

1. **Missing values**: NaN and `?` entries are replaced with the column mode (categorical) or median (continuous).
2. **One-hot encoding**: all 9 categorical columns are expanded via `pd.get_dummies`, producing approximately 30+ binary indicator columns.
3. **Min-max normalisation**: the 7 continuous columns are scaled to [0, 1].
4. **Column alignment**: the `expected_columns` parameter ensures train/test consistency after one-hot encoding.

## Known Limitations and Bias Concerns

- The dataset originates from a specific Portuguese bank and may not generalise to other banking contexts or countries.
- The class distribution is heavily imbalanced (~88% negative). This skew can affect the SVM-X surrogate because most neighbourhood samples may receive the same black-box label, reducing the information available for fitting the local decision boundary.
- The `duration` feature (call duration in seconds) is known to be highly predictive but is only available after the call ends, making it a form of data leakage for real deployment. It is retained here for consistency with the paper's setup.
- This repository uses the Bank dataset strictly for evaluating explanation fidelity, not for making real-world marketing predictions.