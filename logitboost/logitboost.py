"""Implementation of the LogitBoost algorithm."""

import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.base import is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import has_fit_parameter


class LogitBoost(BaseEnsemble, ClassifierMixin, MetaEstimatorMixin):
    """A LogitBoost classifier.

    A LogitBoost [1]_ classifier is a meta-estimator that fits an additive model
    minimizing a logistic loss function.

    Parameters
    ----------
    base_estimator : object, optional
        The base estimator from which the LogitBoost classifier is built. This
        should be a *regressor* with support for sample weighting (by means of a
        `sample_weight` parameter in its `fit()` method). If no `base_estimator`
        is specified, a decision stump is used.

    n_estimators : int, optional
        The number of estimators in the ensemble.

    learning_rate : float, optional
        The learning rate shrinks the contribution of each regressor in the
        ensemble by `learning_rate`.

    References
    ----------
    .. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive
        Logistic Regression: A Statistical View of Boosting". The Annals of
        Statistics. Volume 28, Number 2 (2000), pp. 337--374.
        `JSTOR <https://www.jstor.org/stable/2674028>`__.
        `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
    """
    # Distinct class labels found in the training sample
    classes_: np.ndarray

    # Number of distinct class labels found in the training sample
    n_classes_: int

    # List of lists of estimators trained by the model. The shape is
    # (n_estimators, n_classes)
    estimators_: list

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.):
        super(LogitBoost, self).__init__(base_estimator=base_estimator,
                                         n_estimators=n_estimators)
        self.learning_rate = learning_rate

    def _validate_estimator(self, default=None):
        """Check the estimator and set the `base_estimator_` attribute."""
        # The default regressor for LogitBoost is a decision stump
        default = DecisionTreeRegressor(max_depth=1)
        super(LogitBoost, self)._validate_estimator(default=default)

        if not is_regressor(self.base_estimator_):
            raise ValueError(
                "LogitBoost requires the base estimator to be a regressor.")

        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            estimator_name = self.base_estimator_.__class__.__name__
            raise ValueError(f"{estimator_name} doesn't support sample_weight.")

    def fit(self, X, y, sample_weight=None):
        """Build a LogitBoost classifier from the training data (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training feature data.

        y : array-like of shape = (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), optional
            Initial sample weights. If None, each observation is weighted
            equally.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_estimator()
        X, y, sample_weight = self._validate_fit_params(X, y, sample_weight)

        if self.n_classes_ == 2:
            return self._fit_binary(X, y, sample_weight)
        else:
            return self._fit_multiclass(X, y, sample_weight)

    def _fit_binary(self, X, y, sample_weight):
        """Fit a binary LogitBoost model (Algorithm 3 in Friedman, Hastie, &
        Tibshirani (2000)).
        """
        # TODO
        pass

    def _fit_multiclass(self, X, y, sample_weight):
        """Fit a multiclass LogitBoost model (Algorithm 6 in Friedman, Hastie, &
        Tibshirani (2000)).
        """
        # TODO
        pass

    def _validate_fit_params(self, X, y, sample_weight):
        """Parameter validation for LogitBoost's fit() method."""
        # Validate training data
        X, y = check_X_y(X, y)

        # Convert y to class label indices
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        # Validate sample weight
        if sample_weight is None:
            # Initialize with uniform weights
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = column_or_1d(sample_weight)
            check_consistent_length(X, sample_weight)

            # Check positivity
            sample_weight_sum = sample_weight.sum(dtype=np.float64)
            if np.any(sample_weight < 0) or sample_weight_sum <= 0.:
                raise ValueError("Sample weights must be non-negative and have "
                                 "a positive sum.")

            # Normalize weights
            sample_weight /= sample_weight_sum

        return X, y, sample_weight

    def predict(self, X):
        """TODO
        """
        pass

    def predict_proba(self, X):
        """TODO
        """
        pass

    def decision_function(self, X):
        """TODO
        """
        pass

