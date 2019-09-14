"""Implementation of the LogitBoost algorithm."""

from __future__ import division

import warnings

import numpy as np
from scipy.special import expit, softmax
from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.base import clone, is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (check_X_y, check_is_fitted,
                                      check_random_state)

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
    base_estimator : object, optional (default=None)
        The base estimator from which the LogitBoost classifier is built. This
        should be a *regressor*. If no `base_estimator` is specified, a decision
        stump is used.

    n_estimators : int, optional (default=50)
        The number of estimators per class in the ensemble.

    weight_trim_quantile : float, optional (default=0.05)
        Threshold for weight trimming (see Section 9 in [1]_). The distribution
        of the weights tends to become very skewed in later boosting iterations,
        and the observations with low weights contribute little to the base
        estimator being fitted at that iteration. At each boosting iteration,
        observations with weight smaller than this quantile of the sample weight
        distribution are removed from the data for fitting the base estimator
        (for that iteration only) to speed up computation.

    max_response : float, optional (default=4.0)
        Maximum response value to allow when fitting the base estimators (for
        numerical stability). Values will be clipped to the interval
        [-`max_response`, `max_response`]. See the bottom of p. 352 in [1]_.

    learning_rate : float, optional (default=1.0)
        The learning rate shrinks the contribution of each classifier by
        `learning_rate` during fitting.

    bootstrap : bool, optional (default=False)
        If True, each boosting iteration trains the base estimator using a
        weighted bootstrap sample of the training data. If False, each boosting
        iteration trains the base estimator using the full (weighted) training
        sample. In this case, the base estimator must support sample weighting
        by means of a `sample_weight` parameter in its `fit()` method.

    random_state : int, RandomState instance or None, optional (default=None)
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

    See Also
    --------
    sklearn.tree.DecisionTreeRegressor
        The default base estimator (with `max_depth` = 1).

    References
    ----------
    .. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive
        Logistic Regression: A Statistical View of Boosting". The Annals of
        Statistics. Volume 28, Number 2 (2000), pp. 337--374.
        `JSTOR <https://www.jstor.org/stable/2674028>`__.
        `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
    """

    def __init__(self, base_estimator=None, n_estimators=50,
                 weight_trim_quantile=0.05, max_response=4.0, learning_rate=1.0,
                 bootstrap=False, random_state=None):
        super(LogitBoost, self).__init__(base_estimator=base_estimator,
                                         n_estimators=n_estimators)
        self.weight_trim_quantile = weight_trim_quantile
        self.max_response = max_response
        self.learning_rate = learning_rate
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        """Build a LogitBoost classifier from the training data (`X`, `y`).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training feature data.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        **fit_params : keyword arguments
            Additional keyword arguments to pass to the base estimator's `fit()`
            method.

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

        # Check extra keyword arguments for sample_weight: if the user specifies
        # the sample weight manually, then the boosting iterations will never
        # get to update them themselves
        if "sample_weight" in fit_params:
            warnings.warn("Ignoring user-specified sample_weight.",
                          RuntimeWarning)
            del fit_params["sample_weight"]

        # Delegate actual fitting to helper methods
        if self.n_classes_ == 2:
            self._fit_binary(X, y, random_state, fit_params)
        else:
            self._fit_multiclass(X, y, random_state, fit_params)

        return self

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
            scores = [estimator.predict(X) for estimator in self.estimators_]
            scores = np.sum(scores, axis=0)
        else:
            scores = [[estimator.predict(X) for estimator in estimators]
                      for estimators in self.estimators_]
            scores = np.sum(scores, axis=0).T

        return scores

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
            indices = (scores >= 0).astype(np.int_)
        else:
            indices = scores.argmax(axis=1)

        return self.classes_.take(indices, axis=0)

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
            prob = expit(scores)
            prob = np.column_stack((1 - prob, prob))
        else:
            prob = softmax(scores, axis=1)

        return prob

    def predict_log_proba(self, X):
        """Predict class log-probabilities for `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        log_prob : numpy.ndarray of shape (n_samples, n_classes)
            Array of class log-probabilities of shape (n_samples, n_classes),
            one log-probability for each (input, class) pair.
        """
        return np.log(self.predict_proba(X))

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
        feature).

        Returns
        -------
        feature_importances_ : numpy.ndarray of shape (n_features,)
            The feature importances. Each feature's importance is computed as
            the average feature importance taken over each estimator in the
            trained ensemble. This requires the base estimator to support a
            `feature_importances_` attribute.

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
            raise NotImplementedError("Feature importances is currently only "
                                      "implemented for binary classification "
                                      "tasks.")

        try:
            importances = [estimator.feature_importances_
                           for estimator in self.estimators_]
        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances since base_estimator "
                "does not have a feature_importances_ attribute")

        return np.sum(importances, axis=0) / len(self.estimators_)

    def contributions(self, X, sample_weight=None):
        """Average absolute contribution of each estimator in the ensemble.

        This can be used to compare how much influence different estimators in
        the ensemble have on the final predictions made by the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to average over.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Weights for the samples, for averaging.

        Returns
        -------
        contrib : numpy.ndarray of shape (n_estimators,)
            Average absolute contribution of each estimator in the ensemble.
        """
        check_is_fitted(self, "estimators_")

        if self.n_classes_ == 2:
            predictions = [e.predict(X) for e in self.estimators_]
            predictions = np.abs(predictions)
        else:
            predictions = [[e.predict(X) for e in class_estimators]
                           for class_estimators in self.estimators_]
            predictions = np.abs(predictions)
            predictions = np.mean(predictions, axis=1)

        return np.average(predictions, axis=-1, weights=sample_weight)

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
            scores = 0
            for estimator in self.estimators_:
                scores += estimator.predict(X)
                yield scores
        else:
            scores = 0
            for estimators in self.estimators_:
                new_scores = [estimator.predict(X) for estimator in estimators]
                scores += np.asarray(new_scores).T
                yield scores

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
                indices = (scores > 0).astype(np.int)
                yield self.classes_.take(indices, axis=0)
        else:
            for scores in self.staged_decision_function(X):
                indices = scores.argmax(axis=1)
                yield self.classes_.take(indices, axis=0)

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
                prob = expit(scores)
                yield np.column_stack((1 - prob, prob))
        else:
            for scores in self.staged_decision_function(X):
                yield softmax(scores, axis=1)

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

        sample_weight : array-like of shape (n_samples,) (default=None)
            Weights for the samples.

        Yields
        ------
        accuracy : float
            Accuracy at each stage of boosting.
        """
        for y_pred in self.staged_predict(X):
            yield accuracy_score(y, y_pred, sample_weight=sample_weight)

    def _fit_binary(self, X, y, random_state, fit_params):
        """Fit a binary LogitBoost model.

        This is Algorithm 3 in Friedman, Hastie, & Tibshirani (2000).
        """
        # Initialize with uniform class probabilities
        p = np.full(shape=X.shape[0], fill_value=0.5, dtype=np.float64)

        # Initialize zero scores for each observation
        scores = np.zeros(X.shape[0], dtype=np.float64)

        # Do the boosting iterations to build the ensemble of estimators
        for i in range(self.n_estimators):
            # Update the working response and weights for this iteration
            sample_weight, z = self._weights_and_response(y, p)

            # Create and fit a new base estimator
            X_train, z_train, kwargs = self._boost_fit_args(X, z, sample_weight,
                                                            random_state)
            estimator = self._make_estimator(append=True,
                                             random_state=random_state)
            kwargs.update(fit_params)
            estimator.fit(X_train, z_train, **kwargs)

            # Update the scores and the probability estimates, unless we're
            # doing the last iteration
            if i < self.n_estimators - 1:
                new_scores = estimator.predict(X)
                scores += self.learning_rate * new_scores
                p = expit(scores)

    def _fit_multiclass(self, X, y, random_state, fit_params):
        """Fit a multiclass LogitBoost model.

        This is Algorithm 6 in Friedman, Hastie, & Tibshirani (2000).
        """
        # Initialize with uniform class probabilities
        p = np.full(shape=(X.shape[0], self.n_classes_),
                    fill_value=(1. / self.n_classes_), dtype=np.float64)

        # Initialize zero scores for each observation
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        # Convert y to a one-hot-encoded vector
        y = np.eye(self.n_classes_)[y]

        # Do the boosting iterations to build the ensemble of estimators
        for iboost in range(self.n_estimators):
            # List of estimators for this boosting iteration
            new_estimators = []

            # Create a new estimator for each class
            for j in range(self.n_classes_):
                # Compute the working response and weights
                sample_weight, z = self._weights_and_response(y[:, j], p[:, j])

                # Fit a new base estimator
                X_train, z_train, kwargs = self._boost_fit_args(X, z,
                                                                sample_weight,
                                                                random_state)
                estimator = self._make_estimator(append=False,
                                                 random_state=random_state)
                kwargs.update(fit_params)
                estimator.fit(X_train, z_train, **kwargs)
                new_estimators.append(estimator)

            # Update the scores and the probability estimates
            if iboost < self.n_estimators - 1:
                new_scores = [e.predict(X) for e in new_estimators]
                new_scores = np.asarray(new_scores).T
                new_scores -= new_scores.mean(axis=1, keepdims=True)
                new_scores *= (self.n_classes_ - 1) / self.n_classes_

                scores += self.learning_rate * new_scores
                p = softmax(scores, axis=1)

            self.estimators_.append(new_estimators)

    def _boost_fit_args(self, X, z, sample_weight, random_state):
        """Get arguments to fit a base estimator during boosting."""
        # Ignore observations whose weight is below a quantile threshold
        threshold = np.quantile(sample_weight, self.weight_trim_quantile,
                                interpolation="lower")
        mask = (sample_weight >= threshold)
        X_train = X[mask]
        z_train = z[mask]
        sample_weight = sample_weight[mask]

        if self.bootstrap:
            # Draw a weighted bootstrap sample
            n_samples = X_train.shape[0]
            idx = random_state.choice(n_samples, n_samples, replace=True,
                                      p=(sample_weight / sample_weight.sum()))
            X_train = X[idx]
            z_train = z[idx]
            kwargs = dict()
        else:
            kwargs = dict(sample_weight=sample_weight)

        return X_train, z_train, kwargs

    def _validate_estimator(self, default=None):
        """Check the base estimator and set the `base_estimator_` attribute.

        Parameters
        ----------
        default : scikit-learn estimator, optional (default=None)
            The regressor to use as the base estimator if no `base_estimator`
            __init__() parameter is given. If not specified, this is a
            regression decision stump.
        """
        # The default regressor for LogitBoost is a decision stump
        default = clone(_BASE_ESTIMATOR_DEFAULT) if default is None else default

        super(LogitBoost, self)._validate_estimator(default=default)

        if not is_regressor(self.base_estimator_):
            raise ValueError("LogitBoost requires the base estimator to be a "
                             "regressor.")

    def _weights_and_response(self, y, prob):
        """Update the working weights and response for a boosting iteration."""
        # Samples with very certain probabilities (close to 0 or 1) are weighted
        # less than samples with probabilities closer to 1/2. This is to
        # encourage the higher-uncertainty samples to contribute more during
        # model training
        sample_weight = prob * (1. - prob)

        # Don't allow sample weights to be too close to zero for numerical
        # stability (cf. p. 353 in Friedman, Hastie, & Tibshirani (2000)).
        sample_weight = np.maximum(sample_weight, 2. * _MACHINE_EPSILON)

        # Compute the regression response z = (y - p) / (p * (1 - p))
        with np.errstate(divide="ignore", over="ignore"):
            z = np.where(y, 1. / prob, -1. / (1. - prob))

        # Very negative and very positive values of z are clipped for numerical
        # stability (cf. p. 352 in Friedman, Hastie, & Tibshirani (2000)).
        z = np.clip(z, a_min=-self.max_response, a_max=self.max_response)

        return sample_weight, z
