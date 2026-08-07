# ruff: file-ignore[suspicious-subprocess-import, subprocess-without-shell-equals-true]

import os
import subprocess
import sys
from typing import Any

import pytest

from tensorwaves import configure
from tensorwaves.config import _jax_config, _tensorflow_precision


def _run(code: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def restore_jax_precision():
    import jax

    precision = _jax_config.precision
    x64_enabled = jax.config.x64_enabled
    yield
    _jax_config.precision = precision
    jax.config.update("jax_enable_x64", x64_enabled)


@pytest.mark.parametrize(
    argnames=(
        "environment_value",
        "configuration_value",
        "expected",
    ),
    argvalues=[
        (None, None, True),
        ("false", None, False),
        ("0", None, False),
        ("1", None, True),
        ("true", "float32", False),
    ],
)
def test_jax_precision_configuration(
    environment_value: str | None,
    configuration_value: str | None,
    expected: bool,
):
    environment = os.environ.copy()
    if environment_value is None:
        environment.pop("JAX_ENABLE_X64", None)
    else:
        environment["JAX_ENABLE_X64"] = environment_value
    configuration = (
        ""
        if configuration_value is None
        else f"configure(jax_precision={configuration_value!r})"
    )
    code = f"""
from tensorwaves import configure
from tensorwaves.function._backend import find_function
{configuration}
find_function("array", backend="jax")
import jax
print(jax.config.x64_enabled)
"""
    assert _run(code, env=environment) == str(expected)


def test_jax_reads_environment_variable_set_after_import():
    """JAX resolves ``JAX_ENABLE_X64`` on import, TensorWaves on first backend use."""
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_X64", None)
    code = """
import jax
import os
os.environ["JAX_ENABLE_X64"] = "1"
from tensorwaves.function._backend import find_function
find_function("array", backend="jax")
print(jax.config.x64_enabled)
"""
    assert _run(code, env=environment) == "True"


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_configure_before_creating_arrays(precision: str):
    code = f"""
import jax.numpy as jnp
from tensorwaves import configure
configure(jax_precision={precision!r})
print(jnp.asarray([1.0]).dtype.name)
"""
    assert _run(code) == precision


@pytest.mark.parametrize(
    argnames=("precision", "expected"),
    argvalues=[
        (None, "float64"),
        ("float32", "float32"),
    ],
)
def test_tensorflow_precision_configuration(precision: str | None, expected: str):
    configuration = (
        "" if precision is None else f"configure(tensorflow_precision={precision!r})"
    )
    code = f"""
from tensorwaves import configure
from tensorwaves.data import TFUniformRealNumberGenerator
from tensorwaves.function._backend import find_function
{configuration}
asarray = find_function("asarray", backend="tensorflow")
array = asarray([1.0])
random_values = TFUniformRealNumberGenerator(seed=0)(size=1)
print(array.dtype.name, random_values.dtype.name)
"""
    assert _run(code) == f"{expected} {expected}"


def test_tensorflow_rng_does_not_enable_numpy_behavior():
    code = """
import tensorflow as tf
from tensorwaves.data import TFUniformRealNumberGenerator
TFUniformRealNumberGenerator(seed=0)
print(hasattr(tf.constant([1, 2]), "astype"))
"""
    assert _run(code) == "False"


@pytest.mark.usefixtures("restore_jax_precision")
def test_configure_applies_to_imported_backend():
    import jax

    configure(jax_precision="float32")
    assert not jax.config.x64_enabled
    configure(jax_precision="float64")
    assert jax.config.x64_enabled


@pytest.mark.parametrize(
    argnames=("argument", "message"),
    argvalues=[
        (
            {"jax_precision": "float16"},
            "jax_precision must be 'float32', 'float64', or None",
        ),
        (
            {"tensorflow_precision": "float16"},
            "tensorflow_precision must be 'float32', 'float64', or None",
        ),
    ],
)
def test_configure_precision_value(argument: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        configure(**argument)


def test_configure_validates_before_applying():
    jax_precision = _jax_config.precision
    tensorflow_precision = _tensorflow_precision()
    arguments: dict[str, Any] = {
        "jax_precision": "float32",
        "tensorflow_precision": "float16",
    }
    with pytest.raises(ValueError, match="tensorflow_precision"):
        configure(**arguments)
    assert _jax_config.precision == jax_precision
    assert _tensorflow_precision() == tensorflow_precision
