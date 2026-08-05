from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numba
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # ty: ignore[unresolved-import]

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function import ParametrizedBackendFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    def prange(stop: int) -> range: ...

else:
    from numba import prange


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


def _create_numba_implementations() -> dict[
    str,
    Callable[[np.ndarray, np.ndarray], float],
]:
    @numba.njit(parallel=True)
    def original_unbinned_nll(
        data_intensities: np.ndarray,
        phsp_intensities: np.ndarray,
    ) -> float:
        phsp_sum = 0.0
        for i in prange(len(phsp_intensities)):
            phsp_sum += phsp_intensities[i]
        normalization_factor = len(phsp_intensities) / phsp_sum

        log_likelihood = 0.0
        for i in prange(len(data_intensities)):
            log_likelihood += math.log(normalization_factor * data_intensities[i])
        return -log_likelihood

    @numba.njit(parallel=True)
    def optimized_unbinned_nll(
        data_intensities: np.ndarray,
        phsp_intensities: np.ndarray,
    ) -> float:
        phsp_sum = 0.0
        for i in prange(len(phsp_intensities)):
            phsp_sum += phsp_intensities[i]
        normalization_integral = phsp_sum / len(phsp_intensities)

        log_sum = 0.0
        for i in prange(len(data_intensities)):
            log_sum += math.log(data_intensities[i])
        return len(data_intensities) * math.log(normalization_integral) - log_sum

    return cast(
        "dict[str, Callable[[np.ndarray, np.ndarray], float]]",
        {
            "original": original_unbinned_nll,
            "optimized": optimized_unbinned_nll,
        },
    )


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


def _numpy_intensity(x: np.ndarray, center: float) -> np.ndarray:
    return 1.0 + (x - center) ** 2


@numba.njit(parallel=True)
def _numba_intensity(x: np.ndarray, center: float) -> np.ndarray:
    intensities = np.empty_like(x)
    for i in prange(len(x)):
        intensities[i] = 1.0 + (x[i] - center) ** 2
    return intensities


@jax.jit
def _jax_intensity(x: jax.Array, center: float) -> jax.Array:
    return 1.0 + (x - center) ** 2


@tf.function
def _tensorflow_intensity(x: tf.Tensor, center: float) -> tf.Tensor:
    return 1.0 + (x - center) ** 2


_ESTIMATOR_FUNCTIONS = {
    "numpy": _numpy_intensity,
    "numba": _numba_intensity,
    "jax": _jax_intensity,
    "tensorflow": _tensorflow_intensity,
}


@pytest.fixture(scope="module")
def intensities() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    data_intensities = rng.uniform(low=0.1, high=10.0, size=5_000_000)
    phsp_intensities = rng.uniform(low=0.1, high=10.0, size=1_000_000)
    return data_intensities, phsp_intensities


@pytest.fixture(scope="module")
def estimator_samples() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed=0)
    data = {"x": rng.uniform(low=-2.0, high=2.0, size=1_000_000)}
    phsp = {"x": rng.uniform(low=-2.0, high=2.0, size=1_000_000)}
    return data, phsp


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


@pytest.fixture(scope="module")
def jax_estimator_samples(
    estimator_samples: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    jax.config.update("jax_enable_x64", True)

    data, phsp = estimator_samples
    return (
        {"x": jnp.asarray(data["x"]).block_until_ready()},
        {"x": jnp.asarray(phsp["x"]).block_until_ready()},
    )


@pytest.fixture(scope="module")
def tensorflow_estimator_samples(
    estimator_samples: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    data, phsp = estimator_samples
    return {"x": tnp.asarray(data["x"])}, {"x": tnp.asarray(phsp["x"])}


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


def _create_estimator(
    backend: str,
    data: dict,
    phsp: dict,
) -> UnbinnedNLL:
    function = ParametrizedBackendFunction(
        function=_ESTIMATOR_FUNCTIONS[backend],
        argument_order=("x", "center"),
        parameters={"center": 0.0},
    )
    return UnbinnedNLL(function, data, phsp, backend=backend)


def _compute_estimator_reference(
    data: dict[str, np.ndarray],
    phsp: dict[str, np.ndarray],
    center: float,
) -> float:
    data_intensities = _numpy_intensity(data["x"], center)
    phsp_intensities = _numpy_intensity(phsp["x"], center)
    return _original_unbinned_nll(data_intensities, phsp_intensities)


def _benchmark_estimator_numpy(
    benchmark: Callable[[Callable[[], float]], float],
    backend: str,
    data: dict[str, np.ndarray],
    phsp: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> float:
    estimator = _create_estimator(backend, data, phsp)
    estimator(parameters)

    def run() -> float:
        return estimator(parameters)

    return benchmark(run)


def _benchmark_estimator_jax(
    benchmark: Callable[[Callable[[], jax.Array]], jax.Array],
    data: dict[str, jax.Array],
    phsp: dict[str, jax.Array],
    parameters: dict[str, float],
) -> jax.Array:
    estimator = _create_estimator("jax", data, phsp)
    estimator(parameters).block_until_ready()  # ty: ignore[unresolved-attribute]

    def run() -> jax.Array:
        return estimator(parameters).block_until_ready()  # ty: ignore[unresolved-attribute]

    return benchmark(run)


def _benchmark_estimator_tensorflow(
    benchmark: Callable[[Callable[[], np.ndarray]], np.ndarray],
    data: dict[str, tf.Tensor],
    phsp: dict[str, tf.Tensor],
    parameters: dict[str, float],
) -> np.ndarray:
    estimator = _create_estimator("tensorflow", data, phsp)
    estimator(parameters).numpy()  # ty: ignore[unresolved-attribute]

    def run() -> np.ndarray:
        return estimator(parameters).numpy()  # ty: ignore[unresolved-attribute]

    return benchmark(run)


def _benchmark_numba_implementation(
    benchmark: Callable[[Callable[[], float]], float],
    implementation: str,
    intensities: tuple[np.ndarray, np.ndarray],
) -> float:
    data_intensities, phsp_intensities = intensities
    function = _create_numba_implementations()[implementation]
    function(data_intensities, phsp_intensities)

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
@pytest.mark.parametrize("backend", ["numpy", "numba", "jax", "tensorflow"])
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
    elif backend == "numba":
        result = _benchmark_numba_implementation(
            benchmark,
            implementation,
            intensities,
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


@pytest.mark.benchmark(group="unbinned-nll-estimator")
@pytest.mark.parametrize("backend", ["numpy", "numba", "jax", "tensorflow"])
def test_unbinned_nll_estimator(
    benchmark,
    backend: str,
    estimator_samples: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    request: pytest.FixtureRequest,
) -> None:
    data, phsp = estimator_samples
    parameters = {"center": 0.3}
    reference = _compute_estimator_reference(data, phsp, parameters["center"])

    if backend == "jax":
        jax_data, jax_phsp = request.getfixturevalue("jax_estimator_samples")
        result = _benchmark_estimator_jax(
            benchmark,
            jax_data,
            jax_phsp,
            parameters,
        )
    elif backend == "tensorflow":
        tensorflow_data, tensorflow_phsp = request.getfixturevalue(
            "tensorflow_estimator_samples"
        )
        result = _benchmark_estimator_tensorflow(
            benchmark,
            tensorflow_data,
            tensorflow_phsp,
            parameters,
        )
    else:
        result = _benchmark_estimator_numpy(
            benchmark,
            backend,
            data,
            phsp,
            parameters,
        )

    assert float(np.asarray(result)) == pytest.approx(reference)
