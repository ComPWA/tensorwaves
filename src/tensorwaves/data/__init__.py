# pylint: disable=too-many-arguments
"""The `.data` module takes care of data generation."""

import logging
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.data.phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.interface import (
    DataGenerator,
    DataSample,
    DataTransformer,
    Function,
    RealNumberGenerator,
)

from ._data_sample import (
    finalize_progress_bar,
    get_number_of_events,
    merge_events,
    select_events,
)
from .transform import IdentityTransformer


def generate_data(  # pylint: disable=too-many-arguments too-many-locals
    size: int,
    initial_state_mass: float,
    final_state_masses: Mapping[int, float],
    data_transformer: DataTransformer,
    intensity: Function,
    phsp_generator: Optional[DataGenerator] = None,
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
        phsp_generator = TFPhaseSpaceGenerator(
            initial_state_mass, final_state_masses
        )
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()
    data_generator = IntensityDistributionGenerator(
        domain_generator=phsp_generator,
        function=intensity,
        transformer=data_transformer,
        bunch_size=bunch_size,
    )
    return data_generator.generate(size, random_generator)


def generate_phsp(
    size: int,
    initial_state_mass: float,
    final_state_masses: Mapping[int, float],
    phsp_generator: Optional[DataGenerator] = None,
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
            initial_state_mass, final_state_masses, bunch_size
        )
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()
    return phsp_generator.generate(size, random_generator)


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
    """Generate an hit-and-miss `.DataSample` distribution for a `.Function`.

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
        bunch_size: int = 50_000,
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
        while get_number_of_events(returned_data) < size:
            data_bunch, bunch_max = self._generate_bunch(rng)
            if bunch_max > current_max_intensity:
                current_max_intensity = 1.05 * bunch_max
                if get_number_of_events(returned_data) > 0:
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
                returned_data = merge_events(returned_data, data_bunch)
            else:
                returned_data = data_bunch
            progress_bar.update(
                n=get_number_of_events(returned_data) - progress_bar.n
            )
        finalize_progress_bar(progress_bar)
        return select_events(returned_data, selector=slice(None, size))

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
        hit_and_miss_sample = select_events(
            domain,
            selector=computed_intensities > random_intensities,
        )
        return hit_and_miss_sample, max_intensity
