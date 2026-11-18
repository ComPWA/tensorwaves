# cspell:ignore callspec modifyitems
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _pytest.config import Config as PytestConfig

if TYPE_CHECKING:
    from qrules import ParticleCollection

_TENSORFLOW_PARAMETERS = {"tensorflow", "TFUniformRealNumberGenerator"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests that require TensorFlow if it is not installed.

    Tests that are parametrized over a TensorFlow backend are marked automatically.
    TensorFlow has no wheels for Python 3.14 and higher yet, see
    https://github.com/tensorflow/tensorflow/issues/102890.
    """
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and any(
            _get_name(value) in _TENSORFLOW_PARAMETERS
            for value in callspec.params.values()
        ):
            item.add_marker(pytest.mark.tensorflow)
    if find_spec("tensorflow") is not None:
        return
    skip = pytest.mark.skip(reason="TensorFlow is not installed")
    for item in items:
        if item.get_closest_marker("tensorflow") is not None:
            item.add_marker(skip)


def _get_name(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, type):
        return value.__name__
    return None


@pytest.fixture(scope="session")
def pdg() -> "ParticleCollection":
    from qrules.particle import load_pdg

    return load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig: PytestConfig) -> Path:
    return Path(f"{pytestconfig.rootpath}/tests/output")
