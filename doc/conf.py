# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess


# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree('api', ignore_errors=True)
subprocess.call(
    'sphinx-apidoc '
    '--force '
    '--no-toc '
    '--templatedir _templates '
    '--separate '
    '-o api/ ../tensorwaves/ '
    '../tensorwaves/expertsystem/solvers/constraint; ',
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
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
exclude_patterns = [
    '*build',
    'test',
    'tests',
]
pygments_style = 'sphinx'

todo_include_todos = False
add_module_names = False
viewcode_follow_imported_members = True
autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
html_theme = 'bizstyle'
html_show_sourcelink = False
