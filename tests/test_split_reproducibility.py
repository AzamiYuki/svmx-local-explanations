import numpy as np
from sklearn.model_selection import train_test_split


def _make_synthetic_data(n: int = 120, d: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, 2, size=n)
    return X, y


class TestTrainTestSplitReproducibility:
    def test_same_random_state_gives_same_split(self):
        X, y = _make_synthetic_data()

        split1 = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        split2 = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        X_train_1, X_test_1, y_train_1, y_test_1 = split1
        X_train_2, X_test_2, y_train_2, y_test_2 = split2

        np.testing.assert_array_equal(X_train_1, X_train_2)
        np.testing.assert_array_equal(X_test_1, X_test_2)
        np.testing.assert_array_equal(y_train_1, y_train_2)
        np.testing.assert_array_equal(y_test_1, y_test_2)

    def test_different_random_state_changes_split(self):
        X, y = _make_synthetic_data()

        split1 = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        split2 = train_test_split(
            X, y, test_size=0.25, random_state=7, stratify=y
        )

        X_train_1, X_test_1, y_train_1, y_test_1 = split1
        X_train_2, X_test_2, y_train_2, y_test_2 = split2

        assert not np.array_equal(X_train_1, X_train_2)
        assert not np.array_equal(X_test_1, X_test_2)

    def test_split_shapes_are_consistent(self):
        X, y = _make_synthetic_data(n=200, d=10)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        assert X_train.shape == (160, 10)
        assert X_test.shape == (40, 10)
        assert y_train.shape == (160,)
        assert y_test.shape == (40,)