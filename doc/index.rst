.. title:: Welcome

Welcome to TensorWaves!
=======================

.. list-table::

  * - .. image:: https://readthedocs.org/projects/tensorwaves/badge/?version=latest
        :alt: Documentation build status
        :target: https://tensorwaves.readthedocs.io
      .. image:: https://static.mybinder.org/badge_logo.svg
        :alt: Try out Jupyter notebooks
        :target: https://mybinder.org/v2/gh/ComPWA/tensorwaves/master?filepath=examples%2Fworkflow
      .. image:: https://img.shields.io/badge/License-GPLv3+-blue.svg
        :alt: GPLv3+ license
        :target: https://www.gnu.org/licenses/gpl-3.0-standalone.html
      .. image:: https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod
        :alt: GitPod
        :target: https://gitpod.io/#https://github.com/ComPWA/tensorwaves

  * - .. image:: https://badge.fury.io/py/tensorwaves.svg
        :alt: PyPI package
        :target: https://pypi.org/project/tensorwaves
      .. image:: https://img.shields.io/pypi/pyversions/tensorwaves
        :alt: Supported Python versions
        :target: https://pypi.org/project/tensorwaves
      .. image:: mypy_badge.svg
        :alt: Checked with mypy
        :target: http://mypy-lang.org

  * - .. image:: https://github.com/ComPWA/tensorwaves/workflows/CI/badge.svg
        :alt: CI status
        :target: https://github.com/ComPWA/tensorwaves/actions?query=branch%3Amaster+workflow%3A%22CI%22
      .. image:: https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg
        :alt: Test Coverage
        :target: https://codecov.io/gh/ComPWA/tensorwaves
      .. image:: https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef
        :alt: Codacy Badge
        :target: https://www.codacy.com/gh/ComPWA/tensorwaves

  * - .. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
        :target: https://github.com/pre-commit/pre-commit
        :alt: pre-commit
      .. image:: https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265
        :alt: Code style: Prettier
        :target: https://prettier.io/
      .. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :alt: Code style: black
        :target: https://github.com/psf/black
      .. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
        :alt: Imports: isort
        :target: https://pycqa.github.io/isort

For a quick demo of the TensorWaves workflow, see the `workflow notebooks on
binder
<https://mybinder.org/v2/gh/ComPWA/tensorwaves/master?filepath=examples%2Fworkflow>`_.

TensorWaves is a Python package for doing Partial Wave Analysis with
`TensorFlow <https://www.tensorflow.org/>`_ as computational backend. The
package originates from :mod:`pycompwa`, which did its computations through
(`ComPWA <https://github.com/ComPWA/ComPWA>`_ — ComPWA's function tree is
similar to TensorFlow graphs and can therefore be replaced by the tools that
TensorFlow already offers. The decision to completely migrate ComPWA to
TensorFlow has been further inspired by `TensorFlowAnalysis
<https://gitlab.cern.ch/poluekt/TensorFlowAnalysis>`_/`AmpliTF
<https://github.com/apoluekt/AmpliTF>`_ and `zfit
<https://github.com/zfit/zfit/>`_.


.. toctree::
  :maxdepth: 2
  :hidden:

  install
  usage
  contribute


.. toctree::
  :maxdepth: 1
  :hidden:

  api

TensorWaves API
===============

* :ref:`General Index <genindex>`
* :ref:`Python Modules Index <modindex>`
* :ref:`Search <search>`
