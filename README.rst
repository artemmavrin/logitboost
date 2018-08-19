==========
LogitBoost
==========

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://badge.fury.io/py/logitboost
    :alt: Python Version

.. image:: https://badge.fury.io/py/logitboost.svg
    :target: https://badge.fury.io/py/logitboost
    :alt: Package Version

.. image:: https://readthedocs.org/projects/logitboost/badge/?version=latest
    :target: https://logitboost.readthedocs.io/?badge=latest
    :alt: Documentation Status

This is a Python 3 implementation of the LogitBoost algorithm [1]_ built on top
of `scikit-learn <http://scikit-learn.org>`__.

Website: https://logitboost.readthedocs.io

Installation
------------

The latest version of LogitBoost can be installed directly after cloning from
GitHub.

.. code-block :: shell

  git clone https://github.com/artemmavrin/logitboost.git
  cd survive
  make install

Moreover, LogitBoost is on the
`Python Package Index (PyPI) <https://pypi.org/project/logitboost/>`__, so a
recent version of it can be installed with the
`pip <https://pip.pypa.io/en/stable/>`__ tool.

.. code-block :: shell

  python -m pip install logitboost

Make sure your Python version is at least 3.6 (LogitBoost was developed using
Python 3.7).

References
----------
.. [1] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Additive Logistic
    Regression: A Statistical View of Boosting". The Annals of Statistics.
    Volume 28, Number 2 (2000), pp. 337--374.
    `JSTOR <https://www.jstor.org/stable/2674028>`__.
    `Project Euclid <https://projecteuclid.org/euclid.aos/1016218223>`__.
