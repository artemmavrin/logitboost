PYTHON := python
SETUP := setup.py
SETUPOPTS := -q
DOCS := docs
SPHINXOPTS := '-q -W'
RM := rm -rf

.PHONY: help install dev docs test distribute clean py_info

help:
	@ echo "Usage:\n"
	@ echo "make install   Install the package using Setuptools."
	@ echo "make dev       Install the package for development using pip."
	@ echo "make docs      Generate package documentation using Sphinx"
	@ echo "make test      Run unit tests and check code coverage."
	@ echo "make clean     Remove auxiliary files."

docs: clean
	make -C $(DOCS) html SPHINXOPTS=$(SPHINXOPTS)

install: clean py_info
	$(PYTHON) $(SETUP) $(SETUPOPTS) install

dev: clean py_info
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install --upgrade --editable .[dev]

test: clean py_info
	coverage run -m pytest
	coverage report --show-missing

distribute: clean py_info
	@ $(PYTHON) -m pip install -q -U twine
	$(PYTHON) $(SETUP) $(SETUPOPTS) sdist bdist_wheel
	@ echo "Upload to PyPI using 'twine upload dist/*'"

clean:
	@ $(RM) $(DOCS)/build $(DOCS)/api/generated
	@ $(RM) src/*.egg-info .eggs .pytest_cache .coverage
	@ $(RM) build dist

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"
