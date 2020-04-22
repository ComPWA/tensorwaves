Welcome to TensorWaves!
=======================

*This package is Work-In-Progress and currently unstable.*

TensorWaves is a Python package for doing Partial Wave Analysis with
`TensorFlow <https://www.tensorflow.org/>`_ as computational backend. The
package originates from :mod:`pycompwa` (see in particular the :mod:`expert
system <.expertsystem>`), which did its computations through (`ComPWA
<https://github.com/ComPWA/ComPWA>`_ — ComPWA's function tree is similar to
TensorFlow graphs and can therefore be replaced by the tools that TensorFlow
already offers. The decision to completely migrate ComPWA to TensorFlow has
been further inspired by `TensorFlowAnalysis
<https://gitlab.cern.ch/poluekt/TensorFlowAnalysis>`_/`AmpliTF
<https://github.com/apoluekt/AmpliTF>`_ and `zfit
<https://github.com/zfit/zfit/>`_.


What tensorwaves strives for
----------------------------

Our aim is simple: facilitate doing partial wave analysis with TensorFlow.
Since partial wave analysis requires knowledge from many PWA experts, the
**developer experience** has our highest priority. For this reason, we try to
make as much use of the fact that Python is an easy and flexible language and
that TensorFlow is continuously simplifying its interfaces. The following rules
of thumb may be of help:

- It should be **straightforward to find and implement new formulas**, so class
  hierarchies should only be introduced once necessary.

- **Follow a clean design**, so that it's easy to find one's way around as a
  physicist. Try to reduce dependencies between modules and categorise
  submodules into main modules.

These ideas resonate with the "Zen of Python" (:pep:`20`): keep it simple, keep
it easy to contribute. Physics research is our end goal after all.


.. toctree::
   :maxdepth: 2
   :hidden:

   installation
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
