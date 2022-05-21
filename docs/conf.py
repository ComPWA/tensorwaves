"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import re
import shutil
import subprocess
import sys

import requests
from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------
project = "TensorWaves"
PACKAGE = "tensorwaves"
REPO_NAME = "tensorwaves"
copyright = "2020, ComPWA"  # noqa: A001
author = "Common Partial Wave Analysis"

# https://docs.readthedocs.io/en/stable/builds.html
BRANCH = os.environ.get("READTHEDOCS_VERSION", "stable")
if BRANCH == "latest":
    BRANCH = "main"
if re.match(r"^\d+$", BRANCH):  # PR preview
    BRANCH = "stable"

if os.path.exists(f"../src/{PACKAGE}/version.py"):
    __RELEASE = get_distribution(PACKAGE).version
    version = ".".join(__RELEASE.split(".")[:3])


# -- Fetch logo --------------------------------------------------------------
def fetch_logo(url: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return
    online_content = requests.get(url, allow_redirects=True)
    with open(output_path, "wb") as stream:
        stream.write(online_content.content)


LOGO_PATH = "_static/logo.svg"
try:
    fetch_logo(
        url="https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg",
        output_path=LOGO_PATH,
    )
except requests.exceptions.ConnectionError:
    pass
if os.path.exists(LOGO_PATH):
    html_logo = LOGO_PATH

# -- Generate API ------------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))
from _relink_references import relink_references  # noqa: E402

relink_references()
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            f"../src/{PACKAGE}/",
            f"../src/{PACKAGE}/version.py",
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
if not os.path.exists("tensorflow.inv"):
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
    f"{PACKAGE}.",
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
    "sphinx_codeautolink",
    "sphinx_comments",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx_thebe",
    "sphinx_togglebutton",
]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr*",
    "tests",
]

# General sphinx settings
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "DataSample": "tensorwaves.interface.DataSample",
    "InputType": "tensorwaves.interface.InputType",
    "OutputType": "tensorwaves.interface.OutputType",
    "ParameterValue": "tensorwaves.interface.ParameterValue",
}
autodoc_typehints_format = "short"
codeautolink_concat_default = True
AUTODOC_INSERT_SIGNATURE_LINEBREAKS = False
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_css_files = []
if AUTODOC_INSERT_SIGNATURE_LINEBREAKS:
    html_css_files.append("linebreaks-api.css")
html_favicon = "_static/favicon.ico"
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/ComPWA/{REPO_NAME}",
    "repository_branch": BRANCH,
    "path_to_docs": "docs",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "show_navbar_depth": 2,
    "show_toc_level": 2,
}
html_title = "TensorWaves"
panels_add_bootstrap_css = False  # wider page width with sphinx-panels
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
version_remapping = {
    "matplotlib": {"3.5.1": "3.5.0"},
    "scipy": {"1.7.3": "1.7.1"},
}


def get_version(package_name: str) -> str:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    constraints_path = f"../.constraints/py{python_version}.txt"
    with open(constraints_path) as stream:
        constraints = stream.read()
    for line in constraints.split("\n"):
        line = line.split("#")[0]  # remove comments
        line = line.strip()
        if not line.startswith(package_name):
            continue
        if not line:
            continue
        line_segments = tuple(line.split("=="))
        if len(line_segments) != 2:
            continue
        _, installed_version, *_ = line_segments
        installed_version = installed_version.strip()
        remapped_versions = version_remapping.get(package_name)
        if remapped_versions is not None:
            existing_version = remapped_versions.get(installed_version)
            if existing_version is not None:
                return existing_version
        return installed_version
    return "stable"


def get_minor_version(package_name: str) -> str:
    installed_version = get_version(package_name)
    if installed_version == "stable":
        return installed_version
    matches = re.match(r"^([0-9]+\.[0-9]+).*$", installed_version)
    if matches is None:
        raise ValueError(
            "Could not find documentation for"
            f" {package_name} v{installed_version}"
        )
    return matches[1]


__SCIPY_URL = f"https://docs.scipy.org/doc/scipy-{get_version('scipy')}/"
r = requests.get(__SCIPY_URL)
if r.status_code != 200:
    __SCIPY_URL = "https://docs.scipy.org/doc/scipy"

__TF_URL = f"https://www.tensorflow.org/versions/r{get_minor_version('tensorflow')}/api_docs/python"
r = requests.get(__TF_URL + "/tf")
if r.status_code != 200:
    __TF_URL = "https://www.tensorflow.org/api_docs/python"

intersphinx_mapping = {
    "ampform": (
        f"https://ampform.readthedocs.io/en/{get_version('ampform')}",
        None,
    ),
    "compwa-org": ("https://compwa-org.readthedocs.io", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "iminuit": ("https://iminuit.readthedocs.io/en/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "matplotlib": (
        f"https://matplotlib.org/{get_version('matplotlib')}",
        None,
    ),
    "numpy": (f"https://numpy.org/doc/{get_minor_version('numpy')}", None),
    "pandas": (
        f"https://pandas.pydata.org/pandas-docs/version/{get_version('pandas')}",
        None,
    ),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": (
        f"https://qrules.readthedocs.io/en/{get_version('qrules')}",
        None,
    ),
    "scipy": (__SCIPY_URL, None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorflow": (__TF_URL, "tensorflow.inv"),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for linkcheck
linkcheck_anchors = False
if os.environ.get("CI", False):
    linkcheck_anchors = True
    linkcheck_anchors_ignore = [
        r"pip\-installation\-gpu\-cuda",
    ]

# Settings for myst_nb
nb_execution_timeout = -1
nb_output_stderr = "remove"

nb_execution_mode = "off"
if "EXECUTE_NB" in os.environ:
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    nb_execution_mode = "cache"

# Settings for myst-parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
BINDER_LINK = f"https://mybinder.org/v2/gh/ComPWA/{REPO_NAME}/{BRANCH}?filepath=docs/usage"
myst_substitutions = {
    "branch": BRANCH,
    "run_interactive": f"""
```{{margin}}
Run this notebook [on Binder]({BINDER_LINK}) or
{{ref}}`locally on Jupyter Lab <compwa-org:develop:Jupyter Notebooks>` to
interactively modify the parameters.
```
""",
}
myst_update_mathjax = False

# Settings for sphinx_comments
comments_config = {
    "hypothesis": True,
    "utterances": {
        "repo": f"ComPWA/{REPO_NAME}",
        "issue-term": "pathname",
        "label": "📝 Docs",
    },
}

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
