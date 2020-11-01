"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------
project = "TensorWaves"
copyright = "2020, ComPWA"
author = "Common Partial Wave Analysis"

__release = get_distribution("expertsystem").version
version = ".".join(__release.split(".")[:3])

# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            "../src/tensorwaves/",
            "-o api/",
            "--force",
            "--no-toc",
            "--templatedir _templates",
            "--separate",
        ]
    ),
    shell=True,
)

# -- Convert sphinx object inventory -----------------------------------------
subprocess.call("sphobjinv convert -o zlib tensorflow.txt", shell=True)


# -- General configuration ---------------------------------------------------
master_doc = "index.md"
source_suffix = {
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}

# The master toctree document.
master_doc = "index"
modindex_common_prefix = [
    "tensorwaves.",
]

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_thebe",
    "sphinx_togglebutton",
    "sphinx_panels",
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
    "repository_branch": "stable",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "expand_sections": ["usage"],
    "theme_dev_mode": True,
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
        "https://pwa.readthedocs.io/projects/expertsystem/en/0.6.2",
        None,
    ),
    "iminuit": ("https://iminuit.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org", None),
    "mypy": ("https://mypy.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "pycompwa": ("https://compwa.github.io", None),
    "python": ("https://docs.python.org/3", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "tensorflow.inv",
    ),
    "tox": ("https://tox.readthedocs.io/en/stable", None),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for linkcheck
linkcheck_anchors = False

# Settings for myst_nb
execution_timeout = -1
nb_output_stderr = "remove"
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}
nb_render_priority["doctest"] = nb_render_priority["html"]

jupyter_execute_notebooks = "off"
if (
    "EXECUTE_NB" in os.environ
    or "READTHEDOCS" in os.environ  # PR preview on RTD
):
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    jupyter_execute_notebooks = "force"

# Settings for myst-parser
myst_admonition_enable = True
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}

# -- Visualize dependencies ---------------------------------------------------
if jupyter_execute_notebooks != "off":
    print("Generating module dependency tree...")
    subprocess.call(
        " ".join(
            [
                "HOME=.",  # in case of calling through tox
                "pydeps",
                "../src/tensorwaves",
                "--exclude *._*",  # hide private modules
                "--max-bacon=1",  # hide external dependencies
                "--noshow",
            ]
        ),
        shell=True,
    )
    if os.path.exists("tensorwaves.svg"):
        with open("api/tensorwaves.rst", "a") as stream:
            stream.write("\n.. image:: /tensorwaves.svg")
