"""Implementations of a `.DataGenerator` for four-momentum samples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

    import numpy as np
    import tensorflow as tf

_LOGGER = logging.getLogger(__name__)


class TFPhaseSpaceGenerator(DataGenerator):
    """Implements a phase space generator using tensorflow.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.
        bunch_size: Size of a bunch that is generated during a hit-and-miss iteration.
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
    """Implements a phase space generator **with weights** using tensorflow.

    The weights are provided in the returned `.DataSample` under the key
    :code:`"weights"`.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding masses.

    .. seealso:: :ref:`amplitude-analysis:2.2 Generate intensity-based sample`
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
    ) -> None:
        try:
            import phasespace  # noqa: PLC0415
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
    return tensor.numpy()  # pyright: ignore[reportOptionalCall]
