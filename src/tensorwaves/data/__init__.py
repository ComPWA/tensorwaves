# pylint: disable=too-many-arguments
"""The `.data` module takes care of data generation."""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.data.phasespace import (
    TFUniformRealNumberGenerator,
    TFWeightedPhaseSpaceGenerator,
)
from tensorwaves.interface import (
    DataGenerator,
    DataSample,
    DataTransformer,
    Function,
    RealNumberGenerator,
    WeightedDataGenerator,
)

from .transform import IdentityTransformer


def generate_data(  # pylint: disable=too-many-arguments too-many-locals
    size: int,
    initial_state_mass: float,
    final_state_masses: Mapping[int, float],
    data_transformer: DataTransformer,
    intensity: Function,
    phsp_generator: Optional[WeightedDataGenerator] = None,
    random_generator: Optional[RealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> DataSample:
    """Facade function for creating data samples based on an intensities.

    Args:
        size: Sample size to generate.
        initial_state_mass: See `.TFWeightedPhaseSpaceGenerator`.
        final_state_masses: See `.TFWeightedPhaseSpaceGenerator`.
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
        phsp_gen_instance = TFWeightedPhaseSpaceGenerator(
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
            momentum_pool = _concatenate(momentum_pool, bunch)
        else:
            momentum_pool = bunch
        progress_bar.update(
            n=_get_number_of_events(momentum_pool) - progress_bar.n
        )
    _finalize_progress_bar(progress_bar)
    return {i: values[:size] for i, values in momentum_pool.items()}


def _generate_data_bunch(
    bunch_size: int,
    phsp_generator: WeightedDataGenerator,
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
    phsp_generator: Optional[WeightedDataGenerator] = None,
    random_generator: Optional[RealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> DataSample:
    """Facade function for creating (unweighted) phase space samples.

    Args:
        size: Sample size to generate.
        initial_state_mass: See `.TFWeightedPhaseSpaceGenerator`.
        final_state_masses: See `.TFWeightedPhaseSpaceGenerator`.
        phsp_generator: Class of a phase space generator. Defaults to
            `.TFWeightedPhaseSpaceGenerator`.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    if phsp_generator is None:
        phsp_generator = TFWeightedPhaseSpaceGenerator(
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
        momentum_pool = _stack_events(momentum_pool, bunch)
        progress_bar.update(n=_get_number_of_events(bunch))
    _finalize_progress_bar(progress_bar)
    return {i: values[:size] for i, values in momentum_pool.items()}


def _get_number_of_events(four_momenta: DataSample) -> int:
    if len(four_momenta) == 0:
        return 0
    return len(next(iter(four_momenta.values())))


def _stack_events(sample1: DataSample, sample2: DataSample) -> DataSample:
    if len(sample1) and len(sample2) and set(sample1) != set(sample2):
        raise ValueError(
            "Keys of data sets are not matching", set(sample2), set(sample1)
        )
    if _get_number_of_events(sample1) == 0:
        return sample2
    return {i: np.vstack((array, sample2[i])) for i, array in sample1.items()}


def _select_events(four_momenta: DataSample, selector: Any) -> DataSample:
    return {i: values[selector] for i, values in four_momenta.items()}


def _finalize_progress_bar(progress_bar: tqdm) -> None:
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()


class NumpyDomainGenerator(DataGenerator):
    """Generate a uniform `.DataSample` as a domain for a `.Function`.

    Args:
        boundaries: A mapping of the keys in the `.DataSample` that is to be
            generated. The boundaries have to be a `tuple` of a minimum and
            a maximum value that define the range for each key in the
            `.DataSample`.
    """

    def __init__(self, boundaries: Dict[str, Tuple[float, float]]) -> None:
        self.__boundaries = boundaries

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        return {
            var_name: rng(size, min_value, max_value)
            for var_name, (min_value, max_value) in self.__boundaries.items()
        }


class IntensityDistributionGenerator(DataGenerator):
    """Generate an hit-and-miss `.DataSample` distribution with a `.Function`.

    Args:
        domain_generator: A `.DataGenerator` that can be used to generate a
            **domain** `.DataSample` over which to evaluate the
            :code:`function`.
        function: An **intensity** `.Function` with which the output
            distribution `.DataSample` is generated using a
            :ref:`hit-and-miss strategy <usage/basics:Hit & miss>`.
        transformer: Optional `.DataTransformer` that can convert a generated
            **domain** `.DataSample` to a `.DataSample` that the
            :code:`function` can take as input.
        bunch_size: Size of a bunch that is generated during a hit-and-miss
            iteration.
    """

    def __init__(
        self,
        domain_generator: DataGenerator,
        function: Function,
        transformer: Optional[DataTransformer] = None,
        bunch_size: int = 10_000,
    ) -> None:
        self.__domain_generator = domain_generator
        if transformer is not None:
            self.__transform = transformer
        else:
            self.__transform = IdentityTransformer()
        self.__function = function
        self.__bunch_size = bunch_size

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        progress_bar = tqdm(
            total=size,
            desc="Generating intensity-based sample",
            disable=logging.getLogger().level > logging.WARNING,
        )
        returned_data: DataSample = {}
        current_max_intensity = 0.0
        while _get_number_of_events(returned_data) < size:
            data_bunch, bunch_max = self._generate_bunch(rng)
            if bunch_max > current_max_intensity:
                current_max_intensity = 1.05 * bunch_max
                if _get_number_of_events(returned_data) > 0:
                    logging.info(
                        f"Processed bunch maximum of {bunch_max} is over"
                        f" current maximum {current_max_intensity}. Restarting"
                        " generation!"
                    )
                    returned_data = {}
                    # reset progress bar
                    progress_bar.update(n=-progress_bar.n)
                    continue
            if len(returned_data):
                returned_data = _concatenate(returned_data, data_bunch)
            else:
                returned_data = data_bunch
            progress_bar.update(
                n=_get_number_of_events(returned_data) - progress_bar.n
            )
        return {i: values[:size] for i, values in returned_data.items()}

    def _generate_bunch(
        self, rng: RealNumberGenerator
    ) -> Tuple[DataSample, float]:
        domain = self.__domain_generator.generate(self.__bunch_size, rng)
        transformed_domain = self.__transform(domain)
        computed_intensities = self.__function(transformed_domain)
        max_intensity: float = np.max(computed_intensities)
        random_intensities = rng(
            size=self.__bunch_size, max_value=max_intensity
        )
        hit_and_miss_sample = _select_events(
            domain,
            selector=computed_intensities > random_intensities,
        )
        return hit_and_miss_sample, max_intensity


def _concatenate(sample1: DataSample, sample2: DataSample) -> DataSample:
    if len(sample1) and len(sample2) and set(sample1) != set(sample2):
        raise ValueError(
            "Keys of data sets are not matching", set(sample2), set(sample1)
        )
    if _get_number_of_events(sample1) == 0:
        return sample2
    return {
        i: np.concatenate((array, sample2[i])) for i, array in sample1.items()
    }
