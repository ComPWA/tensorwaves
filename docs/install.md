# Installation

The fastest way of installing this package is through PyPI:

```shell
python3 -m pip install tensorwaves
```

This installs the
[latest, stable release](https://pypi.org/project/tensorwaves) that you can
find on the [`stable`](https://github.com/ComPWA/tensorwaves/tree/stable)
branch. The latest version on the
[`master`](https://github.com/ComPWA/tensorwaves/tree/master) branch can be
installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/tensorwaves@master
```

In that case, however, we highly recommend using the more dynamic,
{ref}`'editable installation' <pwa:develop:Editable installation>` instead.
This goes as follows:

1. Get the source code (see {doc}`pwa:software/git`):

   ```shell
   git clone https://github.com/ComPWA/tensorwaves.git
   cd tensorwaves
   ```

2. **[Recommended]** Create a virtual environment (see
   {ref}`here <pwa:develop:Virtual environment>`).

3. Install the project in
   {ref}`'editable installation' <pwa:develop:Editable installation>`, as well
   as {ref}`additional dependencies <pwa:develop:Additional dependencies>` for
   the developer:

   ```shell
   # pin dependencies first!
   python3 -m pip install -r reqs/PYTHON_VERSION/requirements-dev.txt
   python3 -m pip install -e .
   ```

That's all! Have a look at the {doc}`/usage` page to try out the package, and
see {doc}`pwa:develop` for tips on how to work with this 'editable' developer
setup!
