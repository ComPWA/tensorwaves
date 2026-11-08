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

_Backend = Literal["numpy", "tensorflow", "jax"]
_BACKEND_EXTRAS: dict[str, tuple[str, str]] = {
    "numpy": ("numpy", ""),
    "tensorflow": ("tensorflow", "tf"),
    "jax": ("jax", "jax"),
}


class TFPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator on top of `phasespace <https://phasespace.readthedocs.io>`_.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.
        bunch_size: Size of a bunch that is generated during a hit-and-miss iteration.
        backend: Computational backend that phasespace uses to generate the sample.
            See `TFWeightedPhaseSpaceGenerator`.
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
        bunch_size: int = 50_000,
        *,
        backend: Literal["numpy", "tensorflow", "jax"] = "numpy",
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

        .. seealso:: :doc:`/usage/phase-space`
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
    """Implements a phase space generator **with weights** on top of `phasespace <https://phasespace.readthedocs.io>`_.

    The weights are provided in the returned `.DataSample` under the key
    :code:`"weights"`.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.
        backend: Computational backend that phasespace uses to generate the sample.
            The :code:`"tensorflow"` and :code:`"jax"` backends require the :code:`tf`
            and :code:`jax` extras. The generated arrays are always `numpy` arrays. Any
            `.RealNumberGenerator` can be used. If its stream cannot be used directly
            by the backend, a seed is drawn from it, so the sample stays reproducible,
            but differs from the sample of another backend.

    .. seealso:: :doc:`/usage/phase-space`
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
        *,
        backend: Literal["numpy", "tensorflow", "jax"] = "numpy",
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


class EvtGenPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator using EvtGen through :code:`pyevtgen`.

    Events are unweighted, so no hit-and-miss step is needed and the sample contains no
    :code:`"weights"`. This generator works with any `.RealNumberGenerator`. The
    :code:`rng` only provides a seed for the random engine of EvtGen, so samples are
    reproducible but differ from those of other generators.

    Generation becomes impractically slow beyond about nine final state particles, see
    :code:`pyevtgen.phase_space()`.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.

    .. seealso:: :doc:`/usage/phase-space` and
        :ref:`amplitude-analysis:2.1 Generate phase space sample`
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
    ) -> None:
        try:
            import pyevtgen  # ruff:ignore[import-outside-top-level]
        except ImportError:  # pragma: no cover
            raise_missing_module_error("pyevtgen", extras_require="evtgen")
        self.__phase_space = pyevtgen.phase_space
        self.__initial_state_mass = initial_state_mass
        self.__final_state_masses = dict(sorted(final_state_masses.items()))

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        r"""Generate a `.DataSample` of phase space four-momenta.

        Returns:
            A `.DataSample` of **four-momenta** arrays of shape :math:`n \times 4`.
        """
        seed = int(rng(1, max_value=2**32)[0])
        momenta = self.__phase_space(
            self.__initial_state_mass,
            list(self.__final_state_masses.values()),
            n=size,
            seed=seed,
        )
        return {
            f"p{label}": momenta[:, i]
            for i, label in enumerate(self.__final_state_masses)
        }


def _activate_backend(backend: _Backend) -> None:
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
    rng: RealNumberGenerator, backend: _Backend
) -> int | np.random.Generator | tf.random.Generator:
    if backend == "numpy" and isinstance(rng, NumpyUniformRNG):
        return rng.generator
    if backend == "tensorflow" and isinstance(rng, TFUniformRealNumberGenerator):
        return rng.generator
    return int(rng(1, min_value=0, max_value=2**31 - 1)[0])
