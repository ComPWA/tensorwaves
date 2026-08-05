from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from collections.abc import Callable


def _original_unbinned_nll(
    data_intensities: np.ndarray,
    phsp_intensities: np.ndarray,
) -> float:
    normalization_factor = 1.0 / np.mean(phsp_intensities)
    likelihoods = normalization_factor * data_intensities
    return -np.sum(np.log(likelihoods))


def _optimized_unbinned_nll(
    data_intensities: np.ndarray,
    phsp_intensities: np.ndarray,
) -> float:
    normalization_integral = np.mean(phsp_intensities)
    return len(data_intensities) * np.log(normalization_integral) - np.sum(
        np.log(data_intensities)
    )


def _create_jax_implementations() -> dict[
    str,
    Callable[[jax.Array, jax.Array], jax.Array],
]:
    @jax.jit
    def original_unbinned_nll(
        data_intensities: jax.Array,
        phsp_intensities: jax.Array,
    ) -> jax.Array:
        normalization_factor = 1.0 / jnp.mean(phsp_intensities)
        likelihoods = normalization_factor * data_intensities
        return -jnp.sum(jnp.log(likelihoods))

    @jax.jit
    def optimized_unbinned_nll(
        data_intensities: jax.Array,
        phsp_intensities: jax.Array,
    ) -> jax.Array:
        normalization_integral = jnp.mean(phsp_intensities)
        return len(data_intensities) * jnp.log(normalization_integral) - jnp.sum(
            jnp.log(data_intensities)
        )

    return {
        "original": original_unbinned_nll,
        "optimized": optimized_unbinned_nll,
    }


def _create_tensorflow_implementations() -> dict[
    str,
    Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
]:
    @tf.function
    def original_unbinned_nll(
        data_intensities: tf.Tensor,
        phsp_intensities: tf.Tensor,
    ) -> tf.Tensor:
        normalization_factor = 1.0 / tnp.mean(phsp_intensities)
        likelihoods = normalization_factor * data_intensities
        return -tnp.sum(tnp.log(likelihoods))

    @tf.function
    def optimized_unbinned_nll(
        data_intensities: tf.Tensor,
        phsp_intensities: tf.Tensor,
    ) -> tf.Tensor:
        normalization_integral = tnp.mean(phsp_intensities)
        n_events = data_intensities.shape[0]
        return n_events * tnp.log(normalization_integral) - tnp.sum(
            tnp.log(data_intensities)
        )

    return {
        "original": original_unbinned_nll,
        "optimized": optimized_unbinned_nll,
    }


_IMPLEMENTATIONS: dict[
    str,
    Callable[[np.ndarray, np.ndarray], float],
] = {
    "original": _original_unbinned_nll,
    "optimized": _optimized_unbinned_nll,
}


@pytest.fixture(scope="module")
def intensities() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    data_intensities = rng.uniform(low=0.1, high=10.0, size=5_000_000)
    phsp_intensities = rng.uniform(low=0.1, high=10.0, size=1_000_000)
    return data_intensities, phsp_intensities


@pytest.fixture(scope="module")
def jax_intensities(
    intensities: tuple[np.ndarray, np.ndarray],
) -> tuple[jax.Array, jax.Array]:
    jax.config.update("jax_enable_x64", True)

    data_intensities, phsp_intensities = intensities
    jax_data_intensities = jnp.asarray(data_intensities).block_until_ready()
    jax_phsp_intensities = jnp.asarray(phsp_intensities).block_until_ready()
    return jax_data_intensities, jax_phsp_intensities


@pytest.fixture(scope="module")
def tensorflow_intensities(
    intensities: tuple[np.ndarray, np.ndarray],
) -> tuple[tf.Tensor, tf.Tensor]:
    data_intensities, phsp_intensities = intensities
    return tnp.asarray(data_intensities), tnp.asarray(phsp_intensities)


def _benchmark_numpy_implementation(
    benchmark: Callable[[Callable[[], float]], float],
    implementation: str,
    intensities: tuple[np.ndarray, np.ndarray],
) -> float:
    data_intensities, phsp_intensities = intensities
    function = _IMPLEMENTATIONS[implementation]

    def run() -> float:
        return function(data_intensities, phsp_intensities)

    return benchmark(run)


def _benchmark_jax_implementation(
    benchmark: Callable[[Callable[[], jax.Array]], jax.Array],
    implementation: str,
    intensities: tuple[jax.Array, jax.Array],
) -> jax.Array:
    data_intensities, phsp_intensities = intensities
    function = _create_jax_implementations()[implementation]
    function(data_intensities, phsp_intensities).block_until_ready()

    def run() -> jax.Array:
        return function(data_intensities, phsp_intensities).block_until_ready()

    return benchmark(run)


def _benchmark_tensorflow_implementation(
    benchmark: Callable[[Callable[[], np.ndarray]], np.ndarray],
    implementation: str,
    intensities: tuple[tf.Tensor, tf.Tensor],
) -> np.ndarray:
    data_intensities, phsp_intensities = intensities
    function = _create_tensorflow_implementations()[implementation]
    function(data_intensities, phsp_intensities).numpy()

    def run() -> np.ndarray:
        return function(data_intensities, phsp_intensities).numpy()

    return benchmark(run)


@pytest.mark.benchmark(group="unbinned-nll-normalization")
@pytest.mark.parametrize("backend", ["numpy", "jax", "tensorflow"])
@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_unbinned_nll_normalization_formula(
    benchmark,
    backend: str,
    implementation: str,
    intensities: tuple[np.ndarray, np.ndarray],
    request: pytest.FixtureRequest,
) -> None:
    reference = _original_unbinned_nll(
        *intensities,
    )
    if backend == "jax":
        result = _benchmark_jax_implementation(
            benchmark,
            implementation,
            request.getfixturevalue("jax_intensities"),
        )
    elif backend == "tensorflow":
        result = _benchmark_tensorflow_implementation(
            benchmark,
            implementation,
            request.getfixturevalue("tensorflow_intensities"),
        )
    else:
        result = _benchmark_numpy_implementation(
            benchmark,
            implementation,
            intensities,
        )

    assert float(np.asarray(result)) == pytest.approx(reference)
