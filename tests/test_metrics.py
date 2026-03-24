import numpy as np

from src.svmx.utils.metrics import (
    binary_classification_metrics,
    probability_mse,
)


class TestBinaryClassificationMetrics:
    def test_returns_expected_keys(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        metrics = binary_classification_metrics(y_true, y_pred)

        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_metric_values_are_in_unit_interval(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        metrics = binary_classification_metrics(y_true, y_pred)

        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    def test_perfect_prediction_gives_one(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])

        metrics = binary_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


class TestProbabilityMSE:
    def test_zero_when_arrays_match(self):
        p_true = np.array([[0.1, 0.9], [0.8, 0.2]])
        p_pred = np.array([[0.1, 0.9], [0.8, 0.2]])

        mse = probability_mse(p_true, p_pred)

        assert mse == 0.0

    def test_positive_when_arrays_differ(self):
        p_true = np.array([[0.1, 0.9], [0.8, 0.2]])
        p_pred = np.array([[0.2, 0.8], [0.7, 0.3]])

        mse = probability_mse(p_true, p_pred)

        assert mse > 0.0

    def test_supports_vector_input(self):
        p_true = np.array([0.1, 0.8, 0.4])
        p_pred = np.array([0.2, 0.7, 0.5])

        mse = probability_mse(p_true, p_pred)

        assert isinstance(mse, float)
        assert mse > 0.0