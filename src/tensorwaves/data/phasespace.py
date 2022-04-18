# pylint: disable=import-outside-toplevel
"""Implementations of `.DataGenerator` and `.WeightedDataGenerator`."""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.function._backend import raise_missing_module_error
from tensorwaves.interface import (
    DataGenerator,
    DataSample,
    RealNumberGenerator,
    WeightedDataGenerator,
)

from ._data_sample import (
    finalize_progress_bar,
    get_number_of_events,
    merge_events,
    select_events,
)
from .rng import TFUniformRealNumberGenerator


class TFPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator using tensorflow.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding
            masses.
        bunch_size: Size of a bunch that is generated during a hit-and-miss
            iteration.
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
        bunch_size: int = 50_000,
    ) -> None:
        self.__phsp_generator = TFWeightedPhaseSpaceGenerator(
            initial_state_mass, final_state_masses
        )
        self.__bunch_size = bunch_size
        # https://github.com/ComPWA/tensorwaves/issues/395
        self.show_progress = True

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        r"""Generate a `.DataSample` of phase space four-momenta.

        Returns:
            A `.DataSample` of **four-momenta** arrays of shape
            :math:`n \times 4`.

        .. seealso:: :ref:`amplitude-analysis:2.1 Generate phase space sample`
        """
        progress_bar = tqdm(
            total=size,
            desc="Generating phase space sample",
            disable=not self.show_progress
            or logging.getLogger().level > logging.WARNING,
        )
        momentum_pool: DataSample = {}
        while get_number_of_events(momentum_pool) < size:
            phsp_momenta, weights = self.__phsp_generator.generate(
                self.__bunch_size, rng
            )
            hit_and_miss_randoms = rng(self.__bunch_size)
            bunch = select_events(
                phsp_momenta, selector=weights > hit_and_miss_randoms
            )
            momentum_pool = merge_events(momentum_pool, bunch)
            progress_bar.update(n=get_number_of_events(bunch))
        finalize_progress_bar(progress_bar)
        return select_events(momentum_pool, selector=slice(None, size))


class TFWeightedPhaseSpaceGenerator(WeightedDataGenerator):
    """Implements a phase space generator **with weights** using tensorflow.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding
            masses.

    .. seealso:: :ref:`amplitude-analysis:2.2 Generate intensity-based sample`
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
    ) -> None:
        try:
            import phasespace
        except ImportError:  # pragma: no cover
            raise_missing_module_error("phasespace", extras_require="phsp")

        sorted_ids = sorted(final_state_masses)
        self.__phsp_gen = phasespace.nbody_decay(
            mass_top=initial_state_mass,
            masses=[final_state_masses[i] for i in sorted_ids],
            names=list(map(str, sorted_ids)),
        )

    def generate(
        self, size: int, rng: RealNumberGenerator
    ) -> tuple[DataSample, np.ndarray]:
        r"""Generate a `.DataSample` of phase space four-momenta with weights.

        Returns:
            A `tuple` of a `.DataSample` (**four-momenta**) with an event-wise
            sequence of weights. The four-momenta are arrays of shape
            :math:`n \times 4`.
        """
        if not isinstance(rng, TFUniformRealNumberGenerator):
            raise TypeError(
                f"{type(self).__name__} requires a "
                f"{TFUniformRealNumberGenerator.__name__}, but got a "
                f"{type(rng).__name__}"
            )
        weights, particles = self.__phsp_gen.generate(
            n_events=size, seed=rng.generator
        )
        phsp_momenta = {
            f"p{label}": momenta.numpy()[:, [3, 0, 1, 2]]
            for label, momenta in particles.items()
        }
        return phsp_momenta, weights.numpy()
