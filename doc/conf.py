"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import shutil
import subprocess


# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    "sphinx-apidoc "
    "--force "
    "--no-toc "
    "--templatedir _templates "
    "--separate "
    "-o api/ ../tensorwaves/; ",
    shell=True,
)

# -- Convert sphinx object inventory -----------------------------------------
subprocess.call(
    "sphobjinv convert -o zlib tensorflow.txt", shell=True,
)


# -- Project information -----------------------------------------------------
project = "TensorWaves"
copyright = "2020, ComPWA"
author = "The ComPWA Team"


# -- General configuration ---------------------------------------------------
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
exclude_patterns = [
    "*build",
    "test",
    "tests",
]

# General sphinx settings
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__call__, __eq__",
}
html_copy_source = False  # do not copy rst files
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"
todo_include_todos = False
viewcode_follow_imported_members = True

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing
nitpick_ignore = [
    ("py:class", "tensorflow.keras.losses.Loss"),
    ("py:class", "tensorflow.python.keras.losses.Loss"),
    ("py:obj", "Loss"),
]

# Intersphinx settings
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python/",
        "tensorflow.inv",
    ),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pycompwa": ("https://compwa.github.io/", None),
    "python": ("https://docs.python.org/3", None),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for linkcheck
linkcheck_anchors = False
linkcheck_ignore = [
    "https://pypi.org/project/tensorwaves",
]
