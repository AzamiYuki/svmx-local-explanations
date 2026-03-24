"""Decision Tree target model for binary tabular classification."""

from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Build a Decision Tree classifier with sensible defaults.

    max_depth is left unconstrained to match the paper's setup, which
    uses standard sklearn defaults for each target model.
    """
    return DecisionTreeClassifier(
        random_state=random_state,
    )