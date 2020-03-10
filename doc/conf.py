# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

# Generate indices for the Python modules in the folder above
subprocess.call(
    'sphinx-apidoc -f -M -o api ..',
    shell=True,
)

# -- Project information -----------------------------------------------------
project = 'TensorWaves'
copyright = '2020, ComPWA'
author = 'The ComPWA Team'


# -- Include constructors ----------------------------------------------------
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']
exclude_patterns = ['*build', 'test', 'tests']
pygments_style = 'sphinx'

todo_include_todos = False
add_module_names = False
viewcode_follow_imported_members = True
autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
html_theme = 'nature'
html_show_sourcelink = False
