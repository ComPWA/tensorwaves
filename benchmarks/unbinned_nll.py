from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _original_unbinned_nll(
    bare_intensities: np.ndarray,
    phsp_intensities: np.ndarray,
) -> float:
    normalization_factor = 1.0 / np.mean(phsp_intensities)
    likelihoods = normalization_factor * bare_intensities
    return -np.sum(np.log(likelihoods))


def _optimized_unbinned_nll(
    bare_intensities: np.ndarray,
    phsp_intensities: np.ndarray,
) -> float:
    normalization_integral = np.mean(phsp_intensities)
    return len(bare_intensities) * np.log(normalization_integral) - np.sum(
        np.log(bare_intensities)
    )


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
    bare_intensities = rng.uniform(low=0.1, high=10.0, size=5_000_000)
    phsp_intensities = rng.uniform(low=0.1, high=10.0, size=1_000_000)
    return bare_intensities, phsp_intensities


@pytest.mark.benchmark(group="unbinned-nll-normalization")
@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_unbinned_nll_normalization_formula(
    benchmark,
    implementation: str,
    intensities: tuple[np.ndarray, np.ndarray],
) -> None:
    bare_intensities, phsp_intensities = intensities
    reference = _original_unbinned_nll(
        bare_intensities,
        phsp_intensities,
    )
    result = benchmark(
        _IMPLEMENTATIONS[implementation],
        bare_intensities,
        phsp_intensities,
    )
    assert result == pytest.approx(reference)
