PYTHON := python3
PYTHON2 := python
LINT := pylint
SETUP := setup.py
SETUPOPTS := -q
PACKAGE := logitboost
DOC := doc
RM := rm -rf

.PHONY: help install html test test2 coverage clean lint trim

help:
	@ echo "Usage:"
	@ echo "\tmake install   \t install the package using setuptools."
	@ echo "\tmake html      \t generate documentation using sphinx."
	@ echo "\tmake test      \t run unit tests using pytest."
	@ echo "\tmake test2     \t run unit tests using pytest in Python 2."
	@ echo "\tmake lint      \t check the code using pylint."
	@ echo "\tmake clean     \t remove auxiliary files."

install: clean
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) $(SETUP) $(SETUPOPTS) install

html: clean
	@ mkdir -p $(DOC)/source/_static
	make -C $(DOC) html SPHINXOPTS+='-q -W'

test:
	$(PYTHON) $(SETUP) $(SETUPOPTS) test

test2:
	$(PYTHON2) $(SETUP) $(SETUPOPTS) test

coverage: clean
	coverage run -m pytest
	coverage report

lint:
	$(LINT) $(PACKAGE)

clean: trim clean_doc
	@ $(RM) build dist *.egg-info .eggs .pytest_cache .coverage
	@ find . -name "__pycache__" -type d | xargs rm -rf
	@ find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	@ find . -name "*.pyc" -type f | xargs rm -f
	@ find . -name ".DS_Store" -type f | xargs rm -f
	@ find . -type d -empty -delete

clean_doc:
	@ $(RM) $(DOC)/build $(DOC)/source/generated

# Strip any trailing whitespace from source code
trim:
	@ find $(PACKAGE) -name "*.py" -exec sed -i 's/[[:space:]]*$$//' {} \;
