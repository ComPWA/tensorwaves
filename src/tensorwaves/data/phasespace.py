"""Implementations of a `.DataGenerator` for four-momentum samples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.function._backend import raise_missing_module_error
from tensorwaves.interface import DataGenerator, DataSample, RealNumberGenerator

from ._data_sample import (
    finalize_progress_bar,
    get_number_of_events,
    merge_events,
    select_events,
)
from .rng import NumpyUniformRNG, TFUniformRealNumberGenerator

if TYPE_CHECKING:
    from collections.abc import Mapping

    import tensorflow as tf

_LOGGER = logging.getLogger(__name__)

PhaseSpaceBackend = Literal["numpy", "tensorflow", "jax"]
_BACKEND_EXTRAS: dict[str, tuple[str, str]] = {
    "numpy": ("numpy", ""),
    "tensorflow": ("tensorflow", "tf"),
    "jax": ("jax", "jax"),
}


class TFPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator on top of `phasespace`.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.
        bunch_size: Size of a bunch that is generated during a hit-and-miss iteration.
        backend: Computational backend that `phasespace` uses to generate the sample.
            See `TFWeightedPhaseSpaceGenerator`.
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
        bunch_size: int = 50_000,
        *,
        backend: PhaseSpaceBackend = "numpy",
    ) -> None:
        self.__phsp_generator = TFWeightedPhaseSpaceGenerator(
            initial_state_mass, final_state_masses, backend=backend
        )
        self.__bunch_size = bunch_size
        # https://github.com/ComPWA/tensorwaves/issues/395
        self.show_progress = True

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        r"""Generate a `.DataSample` of phase space four-momenta.

        Returns:
            A `.DataSample` of **four-momenta** arrays of shape :math:`n \times 4`.

        .. seealso:: :ref:`amplitude-analysis:2.1 Generate phase space sample`
        """
        progress_bar = tqdm(
            total=size,
            desc="Generating phase space sample",
            disable=not self.show_progress or _LOGGER.level > logging.WARNING,
        )
        momentum_pool: DataSample = {}
        while get_number_of_events(momentum_pool) < size:
            phsp_momenta = self.__phsp_generator.generate(self.__bunch_size, rng)
            weights = phsp_momenta.get("weights")
            if weights is None:
                msg = (
                    "DataSample returned by"
                    f" {type(self.__phsp_generator).__name__} doesn't contain"
                    ' "weights"'
                )
                raise ValueError(msg)
            hit_and_miss_randoms = rng(self.__bunch_size)
            bunch = select_events(phsp_momenta, selector=weights > hit_and_miss_randoms)
            momentum_pool = merge_events(momentum_pool, bunch)
            progress_bar.update(n=get_number_of_events(bunch))
        finalize_progress_bar(progress_bar)
        phsp = select_events(momentum_pool, selector=slice(None, size))
        if len(phsp) != 0:
            del phsp["weights"]
        return phsp


class TFWeightedPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator **with weights** on top of `phasespace`.

    The weights are provided in the returned `.DataSample` under the key
    :code:`"weights"`.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.
        backend: Computational backend that `phasespace` uses to generate the sample.
            The :code:`"tensorflow"` and :code:`"jax"` backends require the :code:`tf`
            and :code:`jax` extras. The generated arrays are always `numpy` arrays. Any
            `.RealNumberGenerator` can be used. If its stream cannot be used directly
            by the backend, a seed is drawn from it, so the sample stays reproducible,
            but differs from the sample of another backend.

    .. seealso:: :ref:`amplitude-analysis:2.2 Generate intensity-based sample`
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
        *,
        backend: PhaseSpaceBackend = "numpy",
    ) -> None:
        if backend not in _BACKEND_EXTRAS:
            msg = f'Backend must be one of {sorted(_BACKEND_EXTRAS)}, got "{backend}"'
            raise ValueError(msg)
        self.__backend = backend
        _activate_backend(backend)
        import phasespace  # ruff:ignore[import-outside-top-level]

        sorted_ids = sorted(final_state_masses)
        self.__phsp_gen = phasespace.nbody_decay(
            mass_top=initial_state_mass,
            masses=[final_state_masses[i] for i in sorted_ids],
            names=list(map(str, sorted_ids)),
        )

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        r"""Generate a `.DataSample` of phase space four-momenta with weights.

        Returns:
            A `tuple` of a `.DataSample` (**four-momenta**) with an event-wise sequence
            of weights. The four-momenta are arrays of shape :math:`n \times 4`.
        """
        _activate_backend(self.__backend)
        weights, particles = self.__phsp_gen.generate(
            n_events=size, seed=_get_seed(rng, self.__backend)
        )
        phsp_momenta = {
            f"p{label}": np.asarray(momenta)[:, [3, 0, 1, 2]]
            for label, momenta in particles.items()
        }
        return {
            "weights": np.asarray(weights),
            **phsp_momenta,
        }


def _activate_backend(backend: PhaseSpaceBackend) -> None:
    """Select the process-wide `phasespace` backend.

    The previous backend is deliberately not restored afterwards, because each switch
    makes `phasespace` recompile its functions, which would happen for every bunch.
    """
    try:
        import phasespace  # ruff:ignore[import-outside-top-level]
    except ImportError:  # pragma: no cover
        raise_missing_module_error("phasespace", extras_require="phsp")
    try:
        phasespace.set_backend(backend)
    except ImportError:  # pragma: no cover
        module_name, extra = _BACKEND_EXTRAS[backend]
        raise_missing_module_error(module_name, extras_require=extra)


def _get_seed(
    rng: RealNumberGenerator, backend: PhaseSpaceBackend
) -> int | np.random.Generator | tf.random.Generator:
    if backend == "numpy" and isinstance(rng, NumpyUniformRNG):
        return rng.generator
    if backend == "tensorflow" and isinstance(rng, TFUniformRealNumberGenerator):
        return rng.generator
    return int(rng(1, min_value=0, max_value=2**31 - 1)[0])
