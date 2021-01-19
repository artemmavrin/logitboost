"""Unit tests for the LogitBoost class"""

from __future__ import division

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from logitboost import LogitBoost


def test_sklearn_api():
    """Make sure LogitBoost is minimally compliant with scikit-learn's API."""
    check_estimator(LogitBoost())


def _toy_dataset_test(load_func, test_size=(1. / 3), random_state=0,
                      min_score_train=0.9, min_score_test=0.9):
    """Create a classification unit test from a scikit-learn toy dataset."""
    # Fetch the dataset
    data = load_func()
    X = data.data
    y = data.target_names[data.target]

    # Distinct classes
    classes = data.target_names
    n_classes = len(classes)

    # Binary/multiclass classification indicator
    is_binary = (n_classes == 2)

    # Shuffle data and split it into training/testing samples
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y,
                         random_state=random_state)

    for bootstrap in (True, False):
        # Fit a LogitBoost model
        logitboost = LogitBoost(bootstrap=bootstrap, random_state=random_state)
        logitboost.fit(X_train, y_train)

        # Compute accuracy scores and assert minimum accuracy
        score_train = logitboost.score(X_train, y_train)
        score_test = logitboost.score(X_test, y_test)
        assert score_train >= min_score_train, \
            ("Failed with bootstrap=%s: training score %.3f less than %.3f"
             % (bootstrap, score_train, min_score_train))
        assert score_test >= min_score_test, \
            ("Failed with bootstrap=%s: testing score %.3f less than %.3f"
             % (bootstrap, score_test, min_score_test))

        # Get probabilities and the decision function
        predict_proba = logitboost.predict_proba(X_test)
        decision_function = logitboost.decision_function(X_test)

        # predict_proba() should always return (n_samples, n_classes)
        assert predict_proba.shape == (X_test.shape[0], n_classes)

        # decision_function() shape depends on the classification task
        if is_binary:
            assert decision_function.shape == (X_test.shape[0],)
        else:
            assert decision_function.shape == (X_test.shape[0], n_classes)

        # Check that the last item of a staged method is the same as a regular
        # method
        staged_predict = np.asarray(list(logitboost.staged_predict(X_test)))
        staged_predict_proba = \
            np.asarray(list(logitboost.staged_predict_proba(X_test)))
        staged_decision_function = \
            np.asarray(list(logitboost.staged_decision_function(X_test)))
        staged_score = \
            np.asarray(list(logitboost.staged_score(X_test, y_test)))

        np.testing.assert_equal(staged_predict[-1], logitboost.predict(X_test))
        np.testing.assert_almost_equal(staged_predict_proba[-1],
                                       logitboost.predict_proba(X_test))
        np.testing.assert_almost_equal(staged_decision_function[-1],
                                       logitboost.decision_function(X_test))
        np.testing.assert_almost_equal(staged_score[-1],
                                       logitboost.score(X_test, y_test))

        # contributions() should return one non-negative number for each
        # estimator in the ensemble
        contrib = logitboost.contributions(X_train)
        assert contrib.shape == (logitboost.n_estimators,)
        assert np.all(contrib >= 0)


def test_breast_cancer():
    """Simple binary classification on the breast cancer dataset."""
    _toy_dataset_test(load_breast_cancer)


def test_digits():
    """Simple multiclass classification on the handwritten digits dataset."""
    _toy_dataset_test(load_digits)


def test_iris():
    """Simple multiclass classification on the iris dataset."""
    _toy_dataset_test(load_iris)


X_simple = [[1], [2], [3]]
y_simple_binary = np.concatenate(([0], [1] * (len(X_simple) - 1)))
y_simple_multiclass = np.arange(len(X_simple))


def test_bad_base_estimator():
    """Tests for errors raised when the base estimator is bad."""
    # LogitBoost base estimators should be regressors, not classifiers
    base_estimator = DecisionTreeClassifier()
    # Validation is done at fitting, not at initialization
    logitboost = LogitBoost(base_estimator)
    with pytest.raises(ValueError):
        logitboost.fit(X_simple, y_simple_binary)


def test_fit_sample_weight():
    """Check that a warning is raised if sample_weights is passed to fit()."""
    logitboost = LogitBoost()
    with pytest.warns(RuntimeWarning):
        logitboost.fit(X_simple, y_simple_binary,
                       sample_weight=np.ones(len(X_simple)))


def test_feature_importances_():
    """Check that the feature_importances_ attribute behaves as expected."""
    # DecisionTreeRegressor supports feature_importances_
    logitboost = LogitBoost(DecisionTreeRegressor())
    # Binary classification should work
    logitboost.fit(X_simple, y_simple_binary)
    assert logitboost.feature_importances_.shape == (np.shape(X_simple)[1],)

    # Multiclass classification should currently fail
    logitboost.fit(X_simple, y_simple_multiclass)
    with pytest.raises(NotImplementedError):
        _ = logitboost.feature_importances_

    # Ridge doesn't support feature_importances_
    logitboost = LogitBoost(Ridge())
    # Even binary classification shouldn't work
    logitboost.fit(X_simple, y_simple_binary)
    with pytest.raises(AttributeError):
        _ = logitboost.feature_importances_

    # Check that the feature_importance_ attribute identifies bad features
    X, y = load_breast_cancer(return_X_y=True)

    # Add a useless constant feature columns to X: it should be the least
    # important
    X = np.column_stack((X, np.zeros(len(X))))

    logitboost = LogitBoost(random_state=0)
    logitboost.fit(X, y)

    feature_importances = logitboost.feature_importances_
    dummy_importance = feature_importances[-1]
    assert dummy_importance == min(feature_importances)
