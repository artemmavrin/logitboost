============
Installation
============

The :mod:`logitboost` package can be installed using the
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
