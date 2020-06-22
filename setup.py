"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools

DATA_FILES = [
    "particle_list.xml",
    "particle_list.yaml",
]


def long_description():
    """Parse long description from readme."""
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setuptools.setup(
    name="tensorwaves",
    version="0.0-alpha2",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ComPWA/tensorwaves",
    packages=setuptools.find_packages(),
    license="GPLv3 or later",
    python_requires=">=3.6, <3.8",
    install_requires=[
        "amplitf",
        "numpy",
        "progress",
        "pyyaml",
        "tensorflow>=2.1,<2.2",
        "xmltodict",
    ],
    package_data={"tensorwaves": DATA_FILES},
    include_package_data=True,
)
