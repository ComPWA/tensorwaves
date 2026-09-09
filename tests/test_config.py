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


def describe_configure():
    def describe_jax_precision():
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
        def it_overrules_the_environment_variable(
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

        def it_is_resolved_on_first_backend_use_not_on_import():
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
        def it_determines_the_dtype_of_new_arrays(precision: str):
            code = f"""
import jax.numpy as jnp
from tensorwaves import configure
configure(jax_precision={precision!r})
print(jnp.asarray([1.0]).dtype.name)
"""
            assert _run(code) == precision

        @pytest.mark.usefixtures("restore_jax_precision")
        def it_can_be_switched_after_the_backend_has_been_imported():
            import jax

            configure(jax_precision="float32")
            assert not jax.config.x64_enabled
            configure(jax_precision="float64")
            assert jax.config.x64_enabled

    def describe_tensorflow_precision():
        @pytest.mark.parametrize(
            argnames=("precision", "expected"),
            argvalues=[
                (None, "float64"),
                ("float32", "float32"),
            ],
        )
        def it_determines_the_dtype_of_arrays_and_random_values(
            precision: str | None, expected: str
        ):
            configuration = (
                ""
                if precision is None
                else f"configure(tensorflow_precision={precision!r})"
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

        def it_does_not_enable_numpy_behavior_through_the_rng():
            code = """
import tensorflow as tf
from tensorwaves.data import TFUniformRealNumberGenerator
TFUniformRealNumberGenerator(seed=0)
print(hasattr(tf.constant([1, 2]), "astype"))
"""
            assert _run(code) == "False"

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
    def it_rejects_an_unsupported_precision(argument: dict[str, Any], message: str):
        with pytest.raises(ValueError, match=message):
            configure(**argument)

    def it_validates_all_arguments_before_applying_any():
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
