==========
LogitBoost
==========

.. image:: https://img.shields.io/pypi/pyversions/logitboost.svg
    :target: https://pypi.org/project/logitboost/
    :alt: Python Version

.. image:: https://img.shields.io/pypi/v/logitboost.svg
    :target: https://pypi.org/project/logitboost/
    :alt: PyPI Package Version

.. image:: https://img.shields.io/github/last-commit/artemmavrin/logitboost/master
    :target: https://github.com/artemmavrin/logitboost
    :alt: Last Commit

.. image:: https://travis-ci.com/artemmavrin/logitboost.svg?branch=master
    :target: https://travis-ci.com/artemmavrin/logitboost
    :alt: Travis CI Build Status

.. image:: https://codecov.io/gh/artemmavrin/logitboost/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/artemmavrin/logitboost
    :alt: Code Coverage

.. image:: https://readthedocs.org/projects/logitboost/badge/?version=latest
    :target: https://logitboost.readthedocs.io/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/artemmavrin/logitboost.svg
    :target: https://github.com/artemmavrin/logitboost/blob/master/LICENSE
    :alt: GitHub License

This is a Python implementation of the LogitBoost classification algorithm [1]_
built on top of `scikit-learn <http://scikit-learn.org>`__.
It supports both binary and multiclass classification; see the
`examples <examples/index.rst>`__.

This package provides the following class, which can be used out-of-the-box like
any scikit-learn estimator:

.. autosummary::

    ~logitboost.LogitBoost

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    installation
    The LogitBoost Class <logitboost>
    examples/index
    Source Code on GitHub <https://github.com/artemmavrin/logitboost>

References
----------
.. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive Logistic
    Regression: A Statistical View of Boosting". The Annals of Statistics.
    Volume 28, Number 2 (2000), pp. 337--374.
    `JSTOR <https://www.jstor.org/stable/2674028>`__.
    `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
