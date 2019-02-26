==========
LogitBoost
==========

.. image:: https://img.shields.io/pypi/pyversions/logitboost.svg
    :target: https://pypi.org/project/logitboost/
    :alt: Python Version

.. image:: https://img.shields.io/pypi/v/logitboost.svg
    :target: https://pypi.org/project/logitboost/
    :alt: PyPI Package Version

.. image:: https://travis-ci.com/artemmavrin/logitboost.svg?branch=master
    :target: https://travis-ci.com/artemmavrin/logitboost
    :alt: Travis CI Build Status

.. image:: https://ci.appveyor.com/api/projects/status/cpg1e5t4oymy7c11?svg=true
    :target: https://ci.appveyor.com/project/artemmavrin/logitboost
    :alt: AppVeyor Build Status

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

The latest version of LogitBoost can be installed directly after cloning from
GitHub.

.. code-block :: shell

  git clone https://github.com/artemmavrin/logitboost.git
  cd logitboost
  make install

Moreover, LogitBoost is on the
`Python Package Index (PyPI) <https://pypi.org/project/logitboost/>`__, so a
recent version of it can be installed with the
`pip <https://pip.pypa.io/en/stable/>`__ tool.

.. code-block :: shell

  python -m pip install logitboost

This project was developed in Python 3.7, and it is tested to also work with
Python 2.7, 3.4, 3.5, and 3.6.

References
----------
.. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive Logistic
    Regression: A Statistical View of Boosting". The Annals of Statistics.
    Volume 28, Number 2 (2000), pp. 337–374.
    `JSTOR <https://www.jstor.org/stable/2674028>`__.
    `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
