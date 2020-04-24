Get the source code
===================

The `tensorwaves` source code is maintained through `Git
<https://git-scm.com/>`_, so you need to `install Git
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ first. Once
you've done so, navigate to a suitable folder and run:

.. code-block:: shell

  git clone --recurse-submodules git@github.com:ComPWA/tensorwaves.git

After that, there should be a folder called :file:`tensorwaves`. We'll call this
folder the **local repository**. If you navigate into it, you can see it has:

* a `tensorwaves
  <https://github.com/ComPWA/tensorwaves/tree/master/tensorwaves>`_ folder with
  Python source code

* a `setup.py <https://github.com/ComPWA/tensorwaves/blob/master/setup.py>`_
  file with instructions for :ref`setuptools <setuptools>`

* a `requirements.txt
  <https://github.com/ComPWA/tensorwaves/blob/master/requirements.txt>`_ file
  listing the Python dependencies

* a `requirements-dev.txt
  <https://github.com/ComPWA/tensorwaves/blob/master/requirements-dev.txt>`_
  containing several tools for :doc:`contributing </contribute>`

These files will be used in the following steps.

.. warning::

  When new commits are merged into the `master branch of tensorwaves
  <https://github.com/ComPWA/tensorwaves/tree/master>`_, you need to update
  your local copy of the source code.

  .. code-block:: shell

    git checkout master
    git pull

  It's best to have a clean your working tree before you do a :command:`git
  pull`. See :doc:`/contribute` for more info.
