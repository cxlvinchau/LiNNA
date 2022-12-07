# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

sys.path.insert(0, os.path.abspath('../'))

project = 'LiNNA'
copyright = '2022, Calvin Chau, Stefanie Mohr, Jan Křetı́nský'
author = 'Calvin Chau, Stefanie Mohr, Jan Křetı́nský'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': True
}