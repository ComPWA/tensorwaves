"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

# -- Copy example notebooks ---------------------------------------------------
print("Copy example notebook and data files")
PATH_SOURCE = "../examples"
PATH_TARGET = "usage"
FILES_TO_COPY = [
    "workflow/1_create_model.ipynb",
    "workflow/2_generate_data.ipynb",
    "workflow/3_perform_fit.ipynb",
]
shutil.rmtree(PATH_TARGET, ignore_errors=True)
os.makedirs(PATH_TARGET, exist_ok=True)
for file_to_copy in FILES_TO_COPY:
    path_from = os.path.join(PATH_SOURCE, file_to_copy)
    path_to = os.path.join(PATH_TARGET, os.path.basename(file_to_copy))
    print("  copy", path_from, "to", path_to)
    shutil.copyfile(path_from, path_to, follow_symlinks=True)

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
subprocess.call("sphobjinv convert -o zlib tensorflow.txt", shell=True)


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
modindex_common_prefix = [
    "tensorwaves.",
]

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_thebe",
    "sphinx_togglebutton",
]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "tests",
]

# General sphinx settings
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join(
        [
            "__call__",
            "__eq__",
        ]
    ),
}
html_copy_source = True  # needed for download notebook button
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/ComPWA/expertsystem",
    "repository_branch": "master",
    "path_to_docs": "doc",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org/v2/gh/ComPWA/expertsystem/master?filepath=examples",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "expand_sections": ["usage"],
}
html_title = "TensorWaves"
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
        "https://pwa.readthedocs.io/projects/expertsystem/en/0.5.0/",
        None,
    ),
    "iminuit": ("https://iminuit.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pwa": ("https://pwa.readthedocs.io/en/latest/", None),
    "pycompwa": ("https://compwa.github.io/", None),
    "python": ("https://docs.python.org/3", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python/",
        "tensorflow.inv",
    ),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for linkcheck
linkcheck_anchors = False

# Settings for nbsphinx
if "NBSPHINX_EXECUTE" in os.environ:
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    nbsphinx_execute = "always"
else:
    nbsphinx_execute = "never"
nbsphinx_timeout = -1
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Settings for myst-parser
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
