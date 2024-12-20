from __future__ import annotations

import os
import subprocess

import requests
from sphinx_api_relink.helpers import (
    get_branch_name,
    get_execution_mode,
    get_package_version,
    pin,
    pin_minor,
    set_intersphinx_version_remapping,
)


def create_tensorflow_inventory() -> None:
    if os.path.exists("tensorflow.inv"):
        return
    subprocess.check_call(  # noqa: S603
        ("sphobjinv", "convert", "-o", "zlib", "tensorflow.txt"),
    )


def get_scipy_url() -> str:
    url = f"https://docs.scipy.org/doc/scipy-{pin('scipy')}/"
    r = requests.get(url)
    if r.status_code != 200:  # noqa: PLR2004
        return "https://docs.scipy.org/doc/scipy"
    return url


def get_tensorflow_url() -> str:
    url = f"https://www.tensorflow.org/versions/r{pin_minor('tensorflow')}/api_docs/python"
    r = requests.get(url + "/tf")
    if r.status_code != 200:  # noqa: PLR2004
        url = "https://www.tensorflow.org/api_docs/python"
    return url


create_tensorflow_inventory()
set_intersphinx_version_remapping({
    "matplotlib": {"3.5.1": "3.5.0"},
    "scipy": {"1.7.3": "1.7.1"},
})

BRANCH = get_branch_name()
ORGANIZATION = "ComPWA"
PACKAGE = "tensorwaves"
REPO_NAME = "tensorwaves"
REPO_TITLE = "TensorWaves"

BINDER_LINK = f"https://mybinder.org/v2/gh/ComPWA/{REPO_NAME}/{BRANCH}?urlpath=lab"
MATPLOTLIB_VERSION_REMAPPING = {
    "matplotlib": {"3.9.1.post1": "3.9.1"},
}

add_module_names = False
api_github_repo = f"{ORGANIZATION}/{REPO_NAME}"
api_target_substitutions: dict[str, str | tuple[str, str]] = {
    "DataSample": "tensorwaves.interface.DataSample",
    "ParameterValue": "tensorwaves.interface.ParameterValue",
    "Path": "pathlib.Path",
    "np.ndarray": "numpy.ndarray",
    "sp.Expr": "sympy.core.expr.Expr",
    "sp.Symbol": "sympy.core.symbol.Symbol",
}
api_target_types: dict[str, str | tuple[str, str]] = {
    "tensorwaves.interface.DataSample": "obj",
    "tensorwaves.interface.InputType": "obj",
    "tensorwaves.interface.OutputType": "obj",
    "tensorwaves.interface.ParameterValue": "obj",
}
author = "Common Partial Wave Analysis"
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
autosectionlabel_prefix_document = True
codeautolink_concat_default = True
comments_config = {
    "hypothesis": True,
    "utterances": {
        "repo": f"ComPWA/{REPO_NAME}",
        "issue-term": "pathname",
        "label": "📝 Docs",
    },
}
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest
copyright = f"2020, {ORGANIZATION}"
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr*",
    "tests",
]
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_api_relink",
    "sphinx_codeautolink",
    "sphinx_comments",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_thebe",
    "sphinx_togglebutton",
]
generate_apidoc_package_path = f"../src/{PACKAGE}"
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%-d %B %Y"
html_logo = (
    "https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg"
)
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Common Partial Wave Analysis",
            "url": "https://compwa.github.io",
            "icon": "_static/favicon.ico",
            "type": "local",
        },
        {
            "name": "GitHub",
            "url": f"https://github.com/ComPWA/{REPO_NAME}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{PACKAGE}",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Conda",
            "url": f"https://anaconda.org/conda-forge/{PACKAGE}",
            "icon": "https://avatars.githubusercontent.com/u/22454001?s=100",
            "type": "url",
        },
        {
            "name": "Launch on Binder",
            "url": (
                f"https://mybinder.org/v2/gh/ComPWA/{REPO_NAME}/{BRANCH}?urlpath=lab"
            ),
            "icon": "https://mybinder.readthedocs.io/en/latest/_static/favicon.png",
            "type": "url",
        },
        {
            "name": "Launch on Colaboratory",
            "url": f"https://colab.research.google.com/github/ComPWA/{REPO_NAME}/blob/{BRANCH}",
            "icon": "https://avatars.githubusercontent.com/u/33467679?s=100",
            "type": "url",
        },
    ],
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "logo": {"text": REPO_TITLE},
    "path_to_docs": "docs",
    "repository_branch": BRANCH,
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
html_title = REPO_TITLE
intersphinx_mapping = {
    "ampform": (f"https://ampform.readthedocs.io/{pin('ampform')}", None),
    "compwa": ("https://compwa.github.io", None),
    "compwa-report": ("https://compwa.github.io/report", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "iminuit": ("https://scikit-hep.org/iminuit", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "matplotlib": (
        f"https://matplotlib.org/{pin('matplotlib', MATPLOTLIB_VERSION_REMAPPING)}",
        None,
    ),
    "numpy": (f"https://numpy.org/doc/{pin_minor('numpy')}", None),
    "pandas": (f"https://pandas.pydata.org/pandas-docs/version/{pin('pandas')}", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": (f"https://qrules.readthedocs.io/{pin('qrules')}", None),
    "scipy": (get_scipy_url(), None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorflow": (get_tensorflow_url(), "tensorflow.inv"),
}
linkcheck_anchors = bool(os.environ.get("CI"))
linkcheck_anchors_ignore = [
    r"pip\-installation\-gpu\-cuda",
]
linkcheck_ignore = [
    "https://unix.stackexchange.com/a/129144",
]
modindex_common_prefix = [f"{PACKAGE}."]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 4
myst_substitutions = {
    "branch": BRANCH,
    "run_interactive": f"""
```{{margin}}
Run this notebook [on Binder]({BINDER_LINK}) or
{{ref}}`locally on Jupyter Lab <compwa:develop:Jupyter Notebooks>` to interactively
modify the parameters.
```
""",
}
myst_update_mathjax = False
nb_execution_mode = get_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "remove"
nitpick_ignore = [
    ("py:class", "tensorflow.keras.losses.Loss"),
    ("py:class", "tensorflow.python.keras.losses.Loss"),
    ("py:obj", "Loss"),
]
nitpicky = True
primary_domain = "py"
project = REPO_TITLE
pygments_style = "sphinx"
release = get_package_version("tensorwaves")
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
version = get_package_version("tensorwaves")
