"""Unit tests for the LogitBoost class"""

from sklearn.utils.estimator_checks import check_estimator

from logitboost import LogitBoost


def test_sklearn_api():
    check_estimator(LogitBoost)
