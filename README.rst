==========
LogitBoost (No Longer Maintained)
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

.. image:: https://github.com/artemmavrin/logitboost/workflows/LogitBoost%20Python%20package/badge.svg
    :target: https://github.com/artemmavrin/logitboost/actions?query=workflow%3A%22LogitBoost+Python+package%22
    :alt: Build Status

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
`examples <https://logitboost.readthedocs.io/examples/index.html>`__.

This package provides a single class, ``LogitBoost``, which can be used
out-of-the-box like any sciki-learn estimator.

Documentation website: https://logitboost.readthedocs.io

Installation
------------

The ``logitboost`` package can be installed using the
`pip <https://pip.pypa.io/en/stable/>`__ utility. For the latest version,
install directly from the package's
`GitHub page <https://github.com/artemmavrin/logitboost>`__:

.. code-block:: bash

    pip install git+https://github.com/artemmavrin/logitboost.git

Alternatively, install a recent release from the
`Python Package Index (PyPI) <https://pypi.org/project/logitboost>`__:

.. code-block:: bash

    pip install logitboost

**Note.** To install the project for development (e.g., to make changes to the
source code), clone the project repository from GitHub and run :code:`make dev`:

.. code-block:: bash

    git clone https://github.com/artemmavrin/logitboost.git
    cd logitboost
    # Optional but recommended: create a new Python virtual environment first
    make dev

This will additionally install the requirements needed to run tests, check code
coverage, and generate documentation.

This project was developed in Python 3.7, and it is tested to also work with
Python 3.6 and 3.8.

References
----------
.. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive Logistic
    Regression: A Statistical View of Boosting". The Annals of Statistics.
    Volume 28, Number 2 (2000), pp. 337–374.
    `JSTOR <https://www.jstor.org/stable/2674028>`__.
    `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
