# ruff: file-ignore[suspicious-subprocess-import, subprocess-without-shell-equals-true]

import os
import subprocess
import sys
from typing import Any

import pytest

from tensorwaves import configure
from tensorwaves.function._backend import find_function


@pytest.mark.parametrize(
    argnames=(
        "environment_value",
        "configuration_value",
        "expected",
    ),
    argvalues=[
        (None, None, True),
        ("false", None, False),
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
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
    )
    assert output.strip() == str(expected)


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
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        text=True,
    )
    assert output.strip() == f"{expected} {expected}"


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


def test_find_function_jax():
    import jax.numpy as jnp

    assert find_function("array", backend="jax") is jnp.array
    assert find_function("linspace", backend="jax") is jnp.linspace
    assert find_function("mean", backend="jax") is jnp.mean


def test_find_function_numpy():
    import numpy as np

    assert find_function("array", backend="numpy") is np.array
    assert find_function("linspace", backend="numpy") is np.linspace
    assert find_function("log", backend="numpy") is np.log
    assert find_function("mean", backend="numpy") is np.mean
    assert find_function("mean", backend="numba") is np.mean


def test_find_function_tf():
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp  # ty:ignore[unresolved-import]

    assert find_function("array", backend="tf") is tnp.array
    assert find_function("linspace", backend="tf") is tnp.linspace
    assert find_function("mean", backend="tf") is tnp.mean
    assert find_function("Tensor", backend="tf") is tf.Tensor
