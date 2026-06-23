# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Project information

project = 'SHARPIE'
copyright = '2025-2026, Hybrid Intelligence Center'
author = 'SHARPIE Development Team'

release = 'rc1'
version = '1.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinxcontrib_django',
    'sphinx_copybutton',
    'myst_parser',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '../images/logo.png'
html_favicon = '../images/logo.png'
html_theme_options = {
    'body_max_width': 'none'
}

# -- Options for Autodoc generator

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
}

# -- Options for Todo list
todo_include_todos = True

# -- Options for Django autodoc generator
sys.path.insert(0, os.path.abspath("../sharpie"))
django_settings = "sharpie.webserver.server.settings"
autodoc_use_legacy_class_based = True


# -- Options for EPUB output
epub_show_urls = 'footnote'
