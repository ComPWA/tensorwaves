# pylint: disable=too-many-arguments
"""The `.data` module takes care of data generation."""

import logging
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.data.phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.interface import (
    DataSample,
    DataTransformer,
    FourMomentumGenerator,
    Function,
    RealNumberGenerator,
)

from . import phasespace, transform

__all__ = [
    "generate_data",
    "generate_phsp",
    "phasespace",
    "transform",
]


def generate_data(  # pylint: disable=too-many-arguments too-many-locals
    size: int,
    initial_state_mass: float,
    final_state_masses: Mapping[int, float],
    data_transformer: DataTransformer,
    intensity: Function,
    phsp_generator: Optional[FourMomentumGenerator] = None,
    random_generator: Optional[RealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> DataSample:
    """Facade function for creating data samples based on an intensities.

    Args:
        size: Sample size to generate.
        initial_state_mass: See `.TFPhaseSpaceGenerator`.
        final_state_masses: See `.TFPhaseSpaceGenerator`.
        data_transformer: An instance of `.DataTransformer` that is used to
            transform a generated `.DataSample` to a `.DataSample` that can be
            understood by the `.Function`.
        intensity: The intensity `.Function` that will be sampled.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    if phsp_generator is None:
        phsp_gen_instance = TFPhaseSpaceGenerator(
            initial_state_mass, final_state_masses
        )
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size,
        desc="Generating intensity-based sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool: DataSample = {}
    current_max = 0.0
    while _get_number_of_events(momentum_pool) < size:
        bunch, maxvalue = _generate_data_bunch(
            bunch_size,
            phsp_gen_instance,
            random_generator,
            intensity,
            data_transformer,
        )
        if maxvalue > current_max:
            current_max = 1.05 * maxvalue
            if _get_number_of_events(momentum_pool) > 0:
                logging.info(
                    "processed bunch maximum of %s is over current"
                    " maximum %s. Restarting generation!",
                    maxvalue,
                    current_max,
                )
                momentum_pool = {}
                progress_bar.update(n=-progress_bar.n)  # reset progress bar
                continue
        if len(momentum_pool):
            momentum_pool = _concatenate_events(momentum_pool, bunch)
        else:
            momentum_pool = bunch
        progress_bar.update(
            n=_get_number_of_events(momentum_pool) - progress_bar.n
        )
    _finalize_progress_bar(progress_bar)
    return {i: values[:size] for i, values in momentum_pool.items()}


def _generate_data_bunch(
    bunch_size: int,
    phsp_generator: FourMomentumGenerator,
    random_generator: RealNumberGenerator,
    intensity: Function,
    adapter: DataTransformer,
) -> Tuple[DataSample, float]:
    phsp_momenta, weights = phsp_generator.generate(
        bunch_size, random_generator
    )
    data_momenta = adapter(phsp_momenta)
    intensities = intensity(data_momenta)
    maxvalue: float = np.max(intensities)

    uniform_randoms = random_generator(bunch_size, max_value=maxvalue)

    hit_and_miss_sample = _select_events(
        phsp_momenta,
        selector=weights * intensities > uniform_randoms,
    )
    return hit_and_miss_sample, maxvalue


def generate_phsp(
    size: int,
    initial_state_mass: float,
    final_state_masses: Mapping[int, float],
    phsp_generator: Optional[FourMomentumGenerator] = None,
    random_generator: Optional[RealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> DataSample:
    """Facade function for creating (unweighted) phase space samples.

    Args:
        size: Sample size to generate.
        initial_state_mass: See `.TFPhaseSpaceGenerator`.
        final_state_masses: See `.TFPhaseSpaceGenerator`.
        phsp_generator: Class of a phase space generator. Defaults to
            `.TFPhaseSpaceGenerator`.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    if phsp_generator is None:
        phsp_generator = TFPhaseSpaceGenerator(
            initial_state_mass, final_state_masses
        )
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size,
        desc="Generating phase space sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool: DataSample = {}
    while _get_number_of_events(momentum_pool) < size:
        phsp_momenta, weights = phsp_generator.generate(
            bunch_size, random_generator
        )
        hit_and_miss_randoms = random_generator(bunch_size)
        bunch = _select_events(
            phsp_momenta, selector=weights > hit_and_miss_randoms
        )
        momentum_pool = _concatenate_events(momentum_pool, bunch)
        progress_bar.update(n=_get_number_of_events(bunch))
    _finalize_progress_bar(progress_bar)
    return {i: values[:size] for i, values in momentum_pool.items()}


def _get_number_of_events(four_momenta: DataSample) -> int:
    if len(four_momenta) == 0:
        return 0
    return len(next(iter(four_momenta.values())))


def _concatenate_events(
    sample1: DataSample, sample2: DataSample
) -> DataSample:
    if len(sample1) and len(sample2) and set(sample1) != set(sample2):
        raise ValueError(
            "Keys of data sets are not matching", set(sample2), set(sample1)
        )
    if _get_number_of_events(sample1) == 0:
        return sample2
    return {
        i: np.vstack((values, sample2[i])) for i, values in sample1.items()
    }


def _select_events(four_momenta: DataSample, selector: Any) -> DataSample:
    return {i: values[selector] for i, values in four_momenta.items()}


def _finalize_progress_bar(progress_bar: tqdm) -> None:
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()


class NumpyUniformRNG(RealNumberGenerator):
    """Implements a uniform real random number generator using `numpy`."""

    def __init__(self, seed: Optional[float] = None):
        self.seed = seed

    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> np.ndarray:
        return self.generator.uniform(size=size, low=min_value, high=max_value)

    @property
    def seed(self) -> Optional[float]:
        return self.__seed

    @seed.setter
    def seed(self, value: Optional[float]) -> None:
        self.__seed = value
        self.generator: np.random.Generator = np.random.default_rng(
            seed=self.seed
        )
