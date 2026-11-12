"""Implementations of a `.DataGenerator` for four-momentum samples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
from .rng import TFUniformRealNumberGenerator

if TYPE_CHECKING:
    from collections.abc import Mapping

    import tensorflow as tf

_LOGGER = logging.getLogger(__name__)
_MAX_PHASESPACE_WEIGHT = 1.0
"""Upper bound of the normalized weights that `phasespace` generates."""


class TFPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator using tensorflow.

    :param initial_state_mass: Mass of the decaying state.
    :param final_state_masses: Masses keyed by final-state ID.
    :param bunch_size: Number of candidates generated per hit-and-miss iteration.
    :param adaptive_max_weight: Perform the hit-and-miss step against an estimate of the
        maximum weight instead of against the upper bound of the weights that
        ``phasespace`` guarantees. This bound is much larger than the weights that
        actually occur, especially for many final state particles, so the estimate
        accepts far more events per bunch. The maximum is estimated from the generated
        bunches with a margin of 5%. If a bunch contains a weight above the current
        estimate, the estimate is raised and the sample is generated anew, like in
        `.IntensityDistributionGenerator`. The generated distribution stays the same,
        but the events differ from those that are generated without this option for the
        same seed. See :doc:`/usage/phase-space`.
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
        bunch_size: int = 50_000,
        *,
        adaptive_max_weight: bool = False,
    ) -> None:
        self.__phsp_generator = TFWeightedPhaseSpaceGenerator(
            initial_state_mass, final_state_masses
        )
        self.__bunch_size = bunch_size
        self.__adaptive_max_weight = adaptive_max_weight
        # https://github.com/ComPWA/tensorwaves/issues/395
        self.show_progress = True

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        r"""Generate a `.DataSample` of phase space four-momenta.

        Returns:
            A `.DataSample` of **four-momenta** arrays of shape :math:`n \times 4`.

        .. seealso::

            :ref:`amplitude-analysis:2.1 Generate phase space sample` and
            :doc:`/usage/phase-space`
        """
        progress_bar = tqdm(
            total=size,
            desc="Generating phase space sample",
            disable=not self.show_progress or _LOGGER.level > logging.WARNING,
        )
        momentum_pool: DataSample = {}
        max_weight = 0.0
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
            if self.__adaptive_max_weight:
                bunch_max = float(np.max(weights))
                if bunch_max > max_weight:
                    max_weight = min(1.05 * bunch_max, _MAX_PHASESPACE_WEIGHT)
                    if get_number_of_events(momentum_pool) > 0:
                        _LOGGER.info(
                            f"Bunch maximum weight {bunch_max} exceeds the estimated"
                            " maximum weight. Restarting generation!"
                        )
                        momentum_pool = {}
                        progress_bar.update(n=-progress_bar.n)
                hit_and_miss_randoms = rng(self.__bunch_size, max_value=max_weight)
            else:
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
    """Implements a phase space generator **with weights** using tensorflow.

    The weights are provided in the returned `.DataSample` under the key ``"weights"``.

    :param initial_state_mass: Mass of the decaying state.
    :param final_state_masses: Masses keyed by final-state ID.

    .. seealso::

        :ref:`amplitude-analysis:2.2 Generate intensity-based sample`
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
    ) -> None:
        try:
            import phasespace  # ruff:ignore[import-outside-top-level]
        except ImportError:  # pragma: no cover
            raise_missing_module_error("phasespace", extras_require="phsp")
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
        if not isinstance(rng, TFUniformRealNumberGenerator):
            msg = (
                f"{type(self).__name__} requires a"
                f" {TFUniformRealNumberGenerator.__name__}, but got a"
                f" {type(rng).__name__}"
            )
            raise TypeError(msg)
        weights, particles = self.__phsp_gen.generate(n_events=size, seed=rng.generator)
        phsp_momenta = {
            f"p{label}": _to_numpy(momenta)[:, [3, 0, 1, 2]]
            for label, momenta in particles.items()
        }
        return {
            "weights": _to_numpy(weights),
            **phsp_momenta,
        }


def _to_numpy(tensor: tf.Tensor) -> np.ndarray:
    return tensor.numpy()
