"""Configuration file for the Sphinx documentation builder."""

import inspect
import os
import subprocess
import sys

import logitboost

# Project information
project = logitboost.__package__
copyright = logitboost.__copyright__.replace('Copyright', '').strip()
author = logitboost.__author__
version = logitboost.__version__
release = version
url = logitboost.__url__

# General configuration

master_doc = 'index'

# Sphinx extension modules
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.linkcode',
    'nbsphinx',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = []

# Options for the intersphinx extension
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scikit-learn': ('http://scikit-learn.org/stable',
                     (None, './_intersphinx/sklearn-objects.inv')),
}

# Generate autosummary
autosummary_generate = True

# sphinx.ext.linkcode: Try to link to source code on GitHub
REVISION_CMD = ['git', 'rev-parse', '--short', 'HEAD']
try:
    _git_revision = subprocess.check_output(REVISION_CMD).strip()
except (subprocess.CalledProcessError, OSError):
    _git_revision = 'master'
else:
    _git_revision = _git_revision.decode('utf-8')


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    module = info.get('module', None)
    fullname = info.get('fullname', None)
    if not module or not fullname:
        return None
    obj = sys.modules.get(module, None)
    if obj is None:
        return None

    for part in fullname.split('.'):
        if not hasattr(obj, part):
            break
        obj = getattr(obj, part)
        if isinstance(obj, property):
            obj = obj.fget
        if hasattr(obj, '__wrapped__'):
            obj = obj.__wrapped__

    file = inspect.getsourcefile(obj)
    if file is None:
        return None
    file = os.path.relpath(file, start=os.path.dirname(logitboost.__file__))
    source, line_start = inspect.getsourcelines(obj)
    line_end = line_start + len(source) - 1
    filename = f'src/logitboost/{file}#L{line_start}-L{line_end}'
    return f'{url}/blob/{_git_revision}/{filename}'


# nbsphinx config: this is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = rf'''
{{% set docname = env.doc2path(env.docname, base='docs') %}}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        This page was generated from `{{{{ docname }}}} <https://github.com/artemmavrin/{project}/blob/{_git_revision}/{{{{ docname }}}}>`_.
        Interactive online version:
        :raw-html:`<a href="https://mybinder.org/v2/gh/artemmavrin/{project}/{_git_revision}?filepath={{{{ docname }}}}">
        <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`
'''
