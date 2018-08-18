"""Implementation of the LogitBoost algorithm."""

import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.base import is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_random_state
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.validation import check_is_fitted

_MACHINE_EPSILON = np.finfo(np.float64).eps


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
        The number of estimators in the ensemble.

    Other Parameters
    ----------------
    learning_rate : float, optional
        The learning rate shrinks the contribution of each regressor in the
        ensemble by `learning_rate`.

    weight_trim_threshold : float, optional
        Threshold for weight trimming (see Section 9 in [1]_). The distribution
        of the weights tends to become very skewed in later the boosting
        iterations, and the observations with low weights contribute little to
        the base estimator being fitted at that iteration. At each boosting
        iteration, observations with weight smaller than this threshold are
        removed from the data (for that iteration only) to speed up computation.
        If this is None, the threshold is the square root of the machine
        epsilon.

    bootstrap : bool, optional
        If True, each boosting iteration trains the base estimator using a
        weighted bootstrap sample of the training data. If False, each bootstrap
        iteration trains the base estimator using the full (weighted) training
        sample. In this case, the base estimator must support sample weighting
        by means of a `sample_weight` parameter in its `fit()` method.

    random_state : int, RandomState instance or None, optional
        If int, `random_state` is the seed used by the random number generator.
        If :class:`~numpy.random.RandomState` instance, `random_state` is the
        random number generator. If None, the random number generator is the
        :class:`~numpy.random.RandomState instance used by :mod:`numpy.random`.

    z_max : float, optional
        Maximum response value to allow when fitting the base estimators. Values
        will be clipped to the interval [-`z_max`, `z_max`].

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

    # All estimators trained by the model
    estimators_: list

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 weight_trim_threshold=None, bootstrap=False, random_state=None,
                 z_max=4.):
        super(LogitBoost, self).__init__(base_estimator=base_estimator,
                                         n_estimators=n_estimators)
        self.learning_rate = learning_rate
        self.weight_trim_threshold = weight_trim_threshold
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.z_max = z_max

    def fit(self, X, y):
        """Build a LogitBoost classifier from the training data (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training feature data.

        y : array-like of shape = (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate __init__() parameters
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        # Validate training data
        X, y = check_X_y(X, y)

        # Convert y to class label indices
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        # Delegate actual fitting to helper methods
        if self.n_classes_ == 2:
            return self._fit_binary(X, y, random_state)
        else:
            return self._fit_multiclass(X, y, random_state)

    def _validate_estimator(self, default=None):
        """Check the estimator and set the `base_estimator_` attribute."""
        # The default regressor for LogitBoost is a decision stump
        default = DecisionTreeRegressor(max_depth=1)
        super(LogitBoost, self)._validate_estimator(default=default)

        if not is_regressor(self.base_estimator_):
            raise ValueError(
                "LogitBoost requires the base estimator to be a regressor.")

        if (not self.bootstrap and
                not has_fit_parameter(self.base_estimator_, "sample_weight")):
            estimator_name = self.base_estimator_.__class__.__name__
            raise ValueError(f"{estimator_name} doesn't support sample_weight.")

    def _fit_binary(self, X, y, random_state):
        """Fit a binary LogitBoost model (Algorithm 3 in Friedman, Hastie, &
        Tibshirani (2000)).
        """
        # Clear any previous estimators and create a new list of estimators
        self.estimators_ = []

        # Initialize with uniform class probabilities
        prob = np.empty(X.shape[0], dtype=np.float64)
        prob[:] = 0.5

        response = np.zeros(X.shape[0], dtype=np.float64)

        for iboost in range(self.n_estimators):
            # Compute the working response and weights
            sample_weight = np.maximum(prob * (1 - prob), 2 * _MACHINE_EPSILON)
            with np.errstate(divide="ignore", over="ignore"):
                z = np.clip(np.where(y == 1, 1 / prob, -1 / (1 - prob)),
                            a_min=-self.z_max, a_max=self.z_max)

            # Fit the base estimator
            X_train, z_train, sample_weight_train = \
                self._get_training_sample(X, z, sample_weight, random_state)
            estimator = self._make_estimator(append=True,
                                             random_state=random_state)
            if self.bootstrap:
                estimator.fit(X_train, z_train)
            else:
                estimator.fit(X_train, z_train,
                              sample_weight=sample_weight_train)

            # Update the response and the probability estimates
            if iboost < self.n_estimators - 1:
                z_pred = estimator.predict(X)
                response += 0.5 * self.learning_rate * z_pred
                prob = np.exp(response)
                prob /= (prob + np.exp(-response))

        return self

    def _fit_multiclass(self, X, y, random_state):
        """Fit a multiclass LogitBoost model (Algorithm 6 in Friedman, Hastie, &
        Tibshirani (2000)).
        """
        # TODO
        raise NotImplementedError()

    def _get_training_sample(self, X, z, sample_weight, random_state):
        """Get training data for the base estimator at each boosting iteration.
        """
        # Normalize the weights
        sample_weight /= sample_weight.sum()

        if self.bootstrap:
            # Draw a weighted bootstrap sample
            n_samples = X.shape[0]
            ind = random_state.choice(n_samples, n_samples, replace=True,
                                      p=sample_weight)
            X = X[ind]
            z = z[ind]
            sample_weight = None
        else:
            # Perform weight trimming if necessary
            if self.weight_trim_threshold is None:
                weight_trim_threshold = np.sqrt(_MACHINE_EPSILON)
            else:
                weight_trim_threshold = self.weight_trim_threshold

            if weight_trim_threshold > 0:
                mask = (sample_weight > weight_trim_threshold)
                X = X[mask]
                z = z[mask]
                sample_weight = sample_weight[mask]

        return X, z, sample_weight

    def predict(self, X):
        """TODO
        """
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        """TODO
        """
        check_is_fitted(self, "estimators_")
        if self.n_classes_ == 2:
            predictions = \
                (estimator.predict(X) for estimator in self.estimators_)
            return np.fromiter(predictions, dtype=np.float64).sum()
        else:
            raise NotImplementedError()

    def decision_function(self, X):
        """TODO
        """
        check_is_fitted(self, "estimators_")
        if self.n_classes_ == 2:
            predictions = np.asarray([estimator.predict(X) for estimator
                                      in self.estimators_], dtype=np.float64)
            return predictions.sum(axis=0)
        else:
            raise NotImplementedError()

