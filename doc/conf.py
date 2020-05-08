"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess


# -- Copy example notebooks ---------------------------------------------------
print("Copy example notebook files")
PATH_SOURCE = "../examples"
PATH_TARGET = "usage"
EXAMPLE_FOLDERS_TO_COPY = [
    "tools",
]
IGNORED_PATTERNS = shutil.ignore_patterns("*/.ipynb_checkpoints/*",)
for root, dirs, _ in os.walk(PATH_TARGET):
    for directory in dirs:
        path = os.path.join(root, directory)
        print("  remove directory", path)
        shutil.rmtree(path)
for root, dirs, _ in os.walk(PATH_SOURCE):
    for directory in dirs:
        if directory not in EXAMPLE_FOLDERS_TO_COPY:
            continue
        path_from = os.path.join(root, directory)
        path_to = os.path.join(PATH_TARGET, directory)
        print("  copy", path_from, "to", path_to)
        shutil.copytree(
            path_from, path_to, symlinks=True, ignore=IGNORED_PATTERNS,
        )
shutil.copyfile(
    os.path.join(PATH_SOURCE, "intensity-recipe.yaml"),
    os.path.join(PATH_TARGET, "intensity-recipe.yaml"),
    follow_symlinks=True,
)


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
source_suffix = [
    ".rst",
    ".ipynb",
]

# The master toctree document.
master_doc = "index"

extensions = [
    "nbsphinx",
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
    "**.ipynb_checkpoints",
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
always_document_param_types = True
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
    "expertsystem": (
        "https://pwa.readthedocs.io/projects/expertsystem/en/latest/",
        None,
    ),
    "iminuit": ("https://iminuit.readthedocs.io/en/latest/", None),
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

# Settings for nbsphinx
if "NBSPHINX_EXECUTE" in os.environ:
    nbsphinx_execute = "always"
else:
    nbsphinx_execute = "never"
nbsphinx_timeout = -1
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
