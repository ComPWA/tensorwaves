"""
A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools

DATA_FILES = [
    'particle_list.xml',
    'particle_list.yaml',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TensorWaves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ComPWA/tensorwaves",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    install_requires=[
        'numpy',
        'progress',
        'pyyaml',
        'tensorflow==2.1',
        'xmltodict',
    ],
    package_data={'tensorwaves': DATA_FILES},
    include_package_data=True,
)
