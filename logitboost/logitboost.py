"""Implementation of the LogitBoost algorithm."""

from __future__ import division

import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.base import clone, is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (check_X_y, check_is_fitted,
                                      check_random_state, has_fit_parameter)

# The smallest representable 64 bit floating point positive number eps such that
# 1.0 + eps != 1.0
_MACHINE_EPSILON = np.finfo(np.float64).eps

# The default regressor for LogitBoost is a decision stump
_BASE_ESTIMATOR_DEFAULT = DecisionTreeRegressor(max_depth=1)


class LogitBoost(BaseEnsemble, ClassifierMixin, MetaEstimatorMixin):
    """A LogitBoost classifier.

    A LogitBoost [1]_ classifier is a meta-estimator that fits an additive model
    minimizing a logistic loss function.

    Parameters
    ----------
    base_estimator : object, optional
        The base estimator from which the LogitBoost classifier is built. This
        should be a *regressor*. If no `base_estimator` is specified, a decision
        stump is used.

    n_estimators : int, optional
        The number of estimators per class in the ensemble.

    weight_trim_quantile : float, optional
        Threshold for weight trimming (see Section 9 in [1]_). The distribution
        of the weights tends to become very skewed in later boosting iterations,
        and the observations with low weights contribute little to the base
        estimator being fitted at that iteration. At each boosting iteration,
        observations with weight smaller than this quantile of the sample weight
        distribution are removed from the data for fitting the base estimator
        (for that iteration only) to speed up computation.

    z_max : float, optional
        Maximum response value to allow when fitting the base estimators (for
        numerical stability). Values will be clipped to the interval
        [-`z_max`, `z_max`]. See the bottom of p. 352 in [1]_.

    learning_rate : float, optional
        The learning rate shrinks the contribution of each classifier by
        `learning_rate` during fitting.

    bootstrap : bool, optional
        If True, each boosting iteration trains the base estimator using a
        weighted bootstrap sample of the training data. If False, each bootstrap
        iteration trains the base estimator using the full (weighted) training
        sample. In this case, the base estimator must support sample weighting
        by means of a `sample_weight` parameter in its `fit()` method.

    random_state : int, RandomState instance or None, optional
        If :class:`int`, `random_state` is the seed used by the random number
        generator. If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator. If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance used
        by :mod:`numpy.random`.

    Attributes
    ----------
    classes_ : numpy.ndarray
        One-dimensional array of unique class labels extracted from the training
        data target vector during fitting.

    estimators_ : list
        All the estimators in the ensemble after fitting. If the task is binary
        classification, this is a list of `n_estimators` fitted base estimators.
        If the task is multiclass classification, this is a list of
        `n_estimators` lists, each containing one base estimator for each class
        label.

    n_classes_ : int
        Number of classes (length of the `classes_` array). If `n_classes` is 2,
        then the task is binary classification. Otherwise, the task is
        multiclass classification.

    n_features_ : int
        Number of features, inferred during fitting.

    References
    ----------
    .. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive
        Logistic Regression: A Statistical View of Boosting". The Annals of
        Statistics. Volume 28, Number 2 (2000), pp. 337--374.
        `JSTOR <https://www.jstor.org/stable/2674028>`__.
        `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
    """

    def __init__(self, base_estimator=None, n_estimators=50,
                 weight_trim_quantile=0.05, z_max=4., learning_rate=1.,
                 bootstrap=False, random_state=None):
        super(LogitBoost, self).__init__(base_estimator=base_estimator,
                                         n_estimators=n_estimators)
        self.weight_trim_quantile = weight_trim_quantile
        self.z_max = z_max
        self.learning_rate = learning_rate
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _validate_estimator(self, default=None):
        """Check the estimator and set the `base_estimator_` attribute."""
        # The default regressor for LogitBoost is a decision stump
        default = clone(_BASE_ESTIMATOR_DEFAULT)
        super(LogitBoost, self)._validate_estimator(default=default)

        if not is_regressor(self.base_estimator_):
            raise ValueError(
                "LogitBoost requires the base estimator to be a regressor.")

        if (not self.bootstrap and
                not has_fit_parameter(self.base_estimator_, "sample_weight")):
            estimator_name = self.base_estimator_.__class__.__name__
            error_msg = "%s doesn't support sample_weight." % estimator_name
            raise ValueError(error_msg)

    def fit(self, X, y):
        """Build a LogitBoost classifier from the training data (`X`, `y`).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training feature data.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : LogitBoost
            Returns this LogitBoost estimator.
        """
        # Validate __init__() parameters
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        # Validate training data
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Convert y to class label indices
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        # Extract number of features in X
        self.n_features_ = X.shape[1]

        # Clear any previous estimators and create a new list of estimators
        self.estimators_ = []

        # Delegate actual fitting to helper methods
        if self.n_classes_ == 2:
            return self._fit_binary(X, y, random_state)
        else:
            return self._fit_multiclass(X, y, random_state)

    def _fit_binary(self, X, y, random_state):
        """Fit a binary LogitBoost model (Algorithm 3 in Friedman, Hastie, &
        Tibshirani (2000))."""
        # Initialize with uniform class probabilities
        prob = np.empty(X.shape[0], dtype=np.float64)
        prob[:] = 0.5

        # Initialize zero scores for each observation
        scores = np.zeros(X.shape[0], dtype=np.float64)

        # Do the boosting iterations to build the ensemble of estimators
        for iboost in range(self.n_estimators):
            scores, prob = self._boost_binary(iboost, X, y, scores, prob,
                                              random_state)

        return self

    def _fit_multiclass(self, X, y, random_state):
        """Fit a multiclass LogitBoost model (Algorithm 6 in Friedman, Hastie, &
        Tibshirani (2000))."""
        # Initialize with uniform class probabilities
        prob = np.empty((X.shape[0], self.n_classes_), dtype=np.float64)
        prob[:] = 1. / self.n_classes_

        # Initialize zero scores for each observation
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        # Get one-hot-encoded class indicators
        y_hot = np.eye(self.n_classes_)[y]

        # Do the boosting iterations to build the ensemble of estimators
        for iboost in range(self.n_estimators):
            scores, prob = self._boost_multiclass(iboost, X, y_hot, scores,
                                                  prob, random_state)

        return self

    def _boost_binary(self, iboost, X, y, scores, prob, random_state):
        """One boosting iteration for the binary classification case."""
        # Compute the working response and weights
        sample_weight, z = _update_weights_and_response(y, prob, self.z_max)

        # Fit a new base estimator
        X_train, z_train, kwargs = \
            self._boost_fit_args(X, z, sample_weight, random_state)
        estimator = self._make_estimator(append=True,
                                         random_state=random_state)
        estimator.fit(X_train, z_train, **kwargs)

        # Update the scores and the probability estimates
        if iboost < self.n_estimators - 1:
            z_pred = estimator.predict(X)
            scores += self.learning_rate * 0.5 * z_pred
            prob = _binary_prob_from_scores(scores)

        return scores, prob

    def _boost_multiclass(self, iboost, X, y_hot, scores, prob, random_state):
        """One boosting iteration for the multiclass classification case."""
        # List of estimators for this boosting iteration
        estimators_iboost = []

        # Create a new estimator for each class
        for iclass in range(self.n_classes_):
            # Compute the working response and weights
            sample_weight, z = _update_weights_and_response(y_hot[:, iclass],
                                                            prob[:, iclass],
                                                            self.z_max)

            # Fit a new base estimator
            X_train, z_train, kwargs = \
                self._boost_fit_args(X, z, sample_weight, random_state)
            estimator = self._make_estimator(append=False,
                                             random_state=random_state)
            estimator.fit(X_train, z_train, **kwargs)
            estimators_iboost.append(estimator)

        # Update the scores and the probability estimates
        if iboost < self.n_estimators - 1:
            predictions = np.asarray([estimator.predict(X) for estimator in
                                      estimators_iboost], dtype=np.float64).T
            predictions -= predictions.mean(axis=1, keepdims=True)
            predictions *= (self.n_classes_ - 1.) / self.n_classes_

            scores += self.learning_rate * predictions
            prob = _multiclass_prob_from_scores(scores)

        self.estimators_.append(estimators_iboost)
        return scores, prob

    def _boost_fit_args(self, X, z, sample_weight, random_state):
        """Get arguments to fit a base estimator during boosting."""
        if self.bootstrap:
            # Draw a weighted bootstrap sample
            n_samples = X.shape[0]
            ind = random_state.choice(n_samples, n_samples, replace=True,
                                      p=(sample_weight / sample_weight.sum()))
            X_train = X[ind]
            z_train = z[ind]
            kwargs = dict()
        else:
            # Perform weight trimming
            threshold = np.quantile(sample_weight, self.weight_trim_quantile,
                                    interpolation="lower")
            mask = (sample_weight >= threshold)
            X_train = X[mask]
            z_train = z[mask]
            kwargs = dict(sample_weight=sample_weight[mask])

        return X_train, z_train, kwargs

    def staged_score(self, X, y, sample_weight=None):
        """Return staged accuracy scores on the given test data and labels.

        This generator method yields the ensemble accuracy score after each
        iteration of boosting and therefore allows monitoring, such as
        determine the score on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            Weights for the samples.

        Yields
        ------
        accuracy : float
            Accuracy at each stage of boosting.
        """
        for y_pred in self.staged_predict(X):
            yield accuracy_score(y, y_pred, sample_weight=sample_weight)

    def predict(self, X):
        """Predict class labels for `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            Array of predicted class labels, one for each input.
        """
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_.take(indices, axis=0)

    def staged_predict(self, X):
        """Return predictions for `X` at each boosting iteration.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Yields
        ------
        labels : numpy.ndarray of shape (n_samples,)
            Array of predicted class labels, one for each input, at each
            boosting iteration.
        """
        if self.n_classes_ == 2:
            for scores in self.staged_decision_function(X):
                yield self.classes_.take((scores > 0).astype(np.int), axis=0)
        else:
            for scores in self.staged_decision_function(X):
                yield self.classes_.take(scores.argmax(axis=1), axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        prob : numpy.ndarray of shape (n_samples, n_classes)
            Array of class probabilities of shape (n_samples, n_classes), one
            probability for each (input, class) pair.
        """
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            prob = _binary_prob_from_scores(scores)
            return np.column_stack((1 - prob, prob))
        else:
            return _multiclass_prob_from_scores(scores)

    def staged_predict_proba(self, X):
        """Predict class probabilities for `X` at each boosting iteration.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Yields
        ------
        prob : numpy.ndarray of shape (n_samples, n_classes)
            Array of class probabilities of shape (n_samples, n_classes), one
            probability for each (input, class) pair, at each boosting
            iteration.
        """
        if self.n_classes_ == 2:
            for scores in self.staged_decision_function(X):
                prob = _binary_prob_from_scores(scores)
                yield np.column_stack((1 - prob, prob))
        else:
            for scores in self.staged_decision_function(X):
                yield _multiclass_prob_from_scores(scores)

    def decision_function(self, X):
        """Compute the decision function of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        scores : numpy.ndarray of shape (n_samples, k)
            The decision function of the input samples. The order of outputs is
            the same of that of the `classes_` attribute. Binary classification
            is a special cases with `k` = 1, otherwise `k` = `n_classes`. For
            binary classification, positive values indicate class 1 and negative
            values indicate class 0.
        """
        check_is_fitted(self, "estimators_")
        if self.n_classes_ == 2:
            predictions = np.asarray([estimator.predict(X) for estimator
                                      in self.estimators_], dtype=np.float64)
            return predictions.sum(axis=0)
        else:
            predictions = np.asarray(
                [[estimator.predict(X) for estimator in estimators]
                 for estimators in self.estimators_], dtype=np.float64)
            return predictions.sum(axis=0).T

    def staged_decision_function(self, X):
        """Compute decision function of `X` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Yields
        ------
        scores : numpy.ndarray of shape (n_samples, k)
            The decision function of the input samples. The order of outputs is
            the same of that of the `classes_` attribute. Binary classification
            is a special cases with `k` = 1, otherwise `k` = `n_classes`. For
            binary classification, positive values indicate class 1 and negative
            values indicate class 0.
        """
        check_is_fitted(self, "estimators_")

        if self.n_classes_ == 2:
            predictions = 0.
            for estimator in self.estimators_:
                predictions = predictions + estimator.predict(X)
                yield predictions
        else:
            predictions = 0.
            for estimators_iboost in self.estimators_:
                predictions_iboost \
                    = np.asarray([estimator.predict(X) for estimator
                                  in estimators_iboost], dtype=np.float64).T
                predictions = predictions + predictions_iboost
                yield predictions

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
        feature).

        Returns
        -------
        feature_importances_ : numpy.ndarray of shape (n_features,)
            The feature importances.

        Raises
        ------
        AttributeError
            Raised if the base estimator doesn't support a
            `feature_importances_` attribute.

        NotImplementedError
            Raised if the task is multiclass classification: feature importance
            is currently only supported for binary classification.
        """
        check_is_fitted(self, "estimators_")

        if self.n_classes_ != 2:
            raise NotImplementedError(
                "Feature importances is currently only implemented for binary "
                "classification tasks.")

        try:
            return np.sum([estimator.feature_importances_ for estimator
                           in self.estimators_], axis=0) / len(self.estimators_)
        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances since base_estimator "
                "does not have a feature_importances_ attribute")


def _update_weights_and_response(y, prob, z_max):
    """Compute the working weights and response for a boosting iteration."""
    with np.errstate(divide="ignore", over="ignore"):
        z = np.clip(np.where(y == 1, 1. / prob, -1. / (1. - prob)),
                    a_min=-z_max, a_max=z_max)

    sample_weight = np.maximum(prob * (1. - prob), 2. * _MACHINE_EPSILON)

    return sample_weight, z


def _binary_prob_from_scores(scores):
    """Convert a LogitBoost score into a probability (binary case)."""
    exp_scores = np.exp(scores)
    return exp_scores / (exp_scores + np.exp(-scores))


def _multiclass_prob_from_scores(scores):
    """Convert a LogitBoost score into a probability (multiclass case)."""
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
