# pylint: disable=too-many-arguments
"""The `.data` module takes care of data generation."""
from __future__ import annotations

import logging

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.interface import (
    DataGenerator,
    DataSample,
    DataTransformer,
    Function,
    RealNumberGenerator,
    WeightedDataGenerator,
)

from ._data_sample import (
    finalize_progress_bar,
    get_number_of_events,
    merge_events,
    select_events,
)

# pyright: reportUnusedImport=false
from .phasespace import (  # noqa:F401
    TFPhaseSpaceGenerator,
    TFWeightedPhaseSpaceGenerator,
)
from .rng import NumpyUniformRNG, TFUniformRealNumberGenerator  # noqa:F401
from .transform import IdentityTransformer, SympyDataTransformer  # noqa:F401


class NumpyDomainGenerator(DataGenerator):
    """Generate a uniform `.DataSample` as a domain for a `.Function`.

    Args:
        boundaries: A mapping of the keys in the `.DataSample` that is to be
            generated. The boundaries have to be a `tuple` of a minimum and
            a maximum value that define the range for each key in the
            `.DataSample`.
    """

    def __init__(self, boundaries: dict[str, tuple[float, float]]) -> None:
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
        domain_transformer: Optional `.DataTransformer` that can convert a generated
            **domain** `.DataSample` to a `.DataSample` that the
            :code:`function` can take as input.
        bunch_size: Size of a bunch that is generated during a hit-and-miss
            iteration.
    """

    def __init__(
        self,
        domain_generator: DataGenerator | WeightedDataGenerator,
        function: Function,
        domain_transformer: DataTransformer | None = None,
        bunch_size: int = 50_000,
    ) -> None:
        self.__domain_generator = domain_generator
        if domain_transformer is not None:
            self.__domain_transformer = domain_transformer
        else:
            self.__domain_transformer = IdentityTransformer()
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
    ) -> tuple[DataSample, float]:
        domain_generator = self.__domain_generator
        if isinstance(domain_generator, WeightedDataGenerator):
            domain, weights = domain_generator.generate(self.__bunch_size, rng)
        else:
            domain = _generate_without_progress_bar(
                domain_generator, self.__bunch_size, rng
            )
            weights = 1  # type: ignore[assignment]
        transformed_domain = self.__domain_transformer(domain)
        computed_intensities = self.__function(transformed_domain)
        max_intensity: float = np.max(computed_intensities)
        random_intensities = rng(
            size=self.__bunch_size, max_value=max_intensity
        )
        hit_and_miss_sample = select_events(
            domain,
            selector=weights * computed_intensities > random_intensities,
        )
        return hit_and_miss_sample, max_intensity


def _generate_without_progress_bar(
    domain_generator: DataGenerator, bunch_size: int, rng: RealNumberGenerator
) -> DataSample:
    # https://github.com/ComPWA/tensorwaves/issues/395
    show_progress = getattr(domain_generator, "show_progress", None)
    if show_progress:
        domain_generator.show_progress = False  # type: ignore[attr-defined]
    domain = domain_generator.generate(bunch_size, rng)
    if show_progress:
        domain_generator.show_progress = show_progress  # type: ignore[attr-defined]
    return domain
