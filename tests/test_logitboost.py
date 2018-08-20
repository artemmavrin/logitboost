"""Unit tests for the LogitBoost class"""

from __future__ import division

from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from logitboost import LogitBoost


def test_sklearn_api():
    """Make sure LogitBoost is minimally compliant with scikit-learn's API."""
    check_estimator(LogitBoost)


def _toy_dataset_test(load_func, test_size=(1. / 3), random_state=0,
                      min_accuracy_train=0.8, min_accuracy_test=0.8):
    """Create a classification unit test from a scikit-learn toy dataset."""
    # Fetch the dataset
    data = load_func()
    X = data.data
    y = data.target_names[data.target]

    # Shuffle data and split it into training/testing samples
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y,
                         random_state=random_state)

    for bootstrap in (True, False):
        # Fit a LogitBoost model
        logitboost = LogitBoost(bootstrap=bootstrap, random_state=random_state)
        logitboost.fit(X_train, y_train)

        # Compute accuracy scores and assert minimum accuracy
        y_pred_train = logitboost.predict(X_train)
        y_pred_test = logitboost.predict(X_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        assert accuracy_train >= min_accuracy_train
        assert accuracy_test >= min_accuracy_test


def test_breast_cancer():
    """Simple binary classification on the breast cancer dataset."""
    _toy_dataset_test(load_breast_cancer)


def test_digits():
    """Simple multiclass classification on the handwritten digits dataset."""
    _toy_dataset_test(load_digits)


def test_iris():
    """Simple multiclass classification on the iris dataset."""
    _toy_dataset_test(load_iris)
