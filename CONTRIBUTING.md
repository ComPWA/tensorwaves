# Contribute


## Some recommended packages for Python development
- [`pytest`](https://docs.pytest.org/en/latest/): Run `pytest` in the main folder of the repository to run all `test_*.py` files
- [`autopep8`](https://pypi.org/project/autopep8/0.8/): Auto-format your Python code
- [`pylint`](https://www.pylint.org/): Scan your code for naming conventions and proper use of Python
- [`rope`](https://github.com/python-rope/rope): Python refactoring tools
- [`sphinx`](https://www.sphinx-doc.org/): Generate documentation of your Python package
- [`doc8`](https://pypi.org/project/doc8/): A style checker for [reStructuredText](https://docutils.sourceforge.io/docs/ref/rst/introduction.html)

If you have added Conda-Forge as a channel, all can be installed in one go:

```
conda install --file requirements_dev.txt
```

Of course, these packages are also available through `pip install`:

```
pip install -r requirements_dev.txt
```


## Conventions
Try to keep test coverage high. You can test current coverage by running

```bash
cd tests
pytest
```

Note that we navigated into the [`tests` directory](./tests) first as to avoid testing the files in the [`tensorwaves` source code directory](./tensorwaves). You can view the coverage report by opening `htmlcov/index.html`.

### Git
  - Please use [conventional commit messages](https://www.conventionalcommits.org/): start the commit with a semantic keyword (see e.g. [Angular](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#type) or [these examples](https://seesparkbox.com/foundry/semantic_commit_messages), followed by [a column](https://git-scm.com/docs/git-interpret-trailers), then the message. The message itself should be in imperative mood—just imagine the commit to give a command to the code framework. So for instance: `feat: add coverage report tools` or `fix: remove `.
  - In the master branch, each commit should compile and be tested. In your own branches, it is recommended to commit frequently (WIP keyword), but squash those commits upon submitting a merge request.

### Python

* Follow [PEP8 conventions](https://www.python.org/dev/peps/pep-0008/).

* Any Python file that's part of a module should contain (in this order):
    1. A docstring describing what the file contains and does, followed by two empty lines.
    2. A definition of [`__all__`](https://docs.python.org/3/tutorial/modules.html#importing-from-a-package) so that you can see immediately what this Python file defines, followed by two empty lines.
    3. Only after these come the `import` statements, following the [PEP8 conventions for imports](https://www.python.org/dev/peps/pep-0008/#imports).

* When calling or defining multiple arguments of a function and multiple entries in a data container, split the entries over multiple lines and end the last entry with a comma, like so:
  ```python
  __all__ = [
     'core',
     'optimizer',
     'physics',
     'plot',
  ]
  ```
  This is to facilitate eventual diff comparisons in Git.
