"""The `.data` module takes care of data generation."""

import logging
from typing import Optional, Tuple

import numpy as np
from ampform.data import EventCollection
from ampform.kinematics import ReactionInfo
from tqdm.auto import tqdm

from tensorwaves.data.phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.interfaces import (
    DataTransformer,
    Function,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)

from . import phasespace, transform

__all__ = [
    "generate_data",
    "generate_phsp",
    "phasespace",
    "transform",
]


def _generate_data_bunch(
    bunch_size: int,
    phsp_generator: PhaseSpaceGenerator,
    random_generator: UniformRealNumberGenerator,
    intensity: Function,
    kinematics: DataTransformer,
) -> Tuple[EventCollection, float]:
    phsp_sample, weights = phsp_generator.generate(
        bunch_size, random_generator
    )
    momentum_pool = EventCollection(phsp_sample)
    dataset = kinematics.transform(momentum_pool)
    intensities = intensity(dataset)
    maxvalue: float = np.max(intensities)

    uniform_randoms = random_generator(bunch_size, max_value=maxvalue)

    hit_and_miss_sample = momentum_pool.select_events(
        weights * intensities > uniform_randoms
    )
    return hit_and_miss_sample, maxvalue


def generate_data(
    size: int,
    reaction_info: ReactionInfo,
    data_transformer: DataTransformer,
    intensity: Function,
    phsp_generator: Optional[PhaseSpaceGenerator] = None,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> EventCollection:
    """Facade function for creating data samples based on an intensities.

    Args:
        size: Sample size to generate.
        reaction_info: Reaction info that is needed to define the phase space.
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
        phsp_gen_instance = TFPhaseSpaceGenerator()
    phsp_gen_instance.setup(reaction_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size,
        desc="Generating intensity-based sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool = EventCollection({})
    current_max = 0.0
    while momentum_pool.n_events < size:
        bunch, maxvalue = _generate_data_bunch(
            bunch_size,
            phsp_gen_instance,
            random_generator,
            intensity,
            data_transformer,
        )
        if maxvalue > current_max:
            current_max = 1.05 * maxvalue
            if momentum_pool.n_events > 0:
                logging.info(
                    "processed bunch maximum of %s is over current"
                    " maximum %s. Restarting generation!",
                    maxvalue,
                    current_max,
                )
                momentum_pool = EventCollection({})
                progress_bar.update(n=-progress_bar.n)  # reset progress bar
                continue
        if np.size(momentum_pool, 0) > 0:
            momentum_pool.append(bunch)
        else:
            momentum_pool = bunch
        progress_bar.update(n=momentum_pool.n_events - progress_bar.n)
    finalize_progress_bar(progress_bar)
    return momentum_pool.select_events(slice(0, size))


def generate_phsp(
    size: int,
    reaction_info: ReactionInfo,
    phsp_generator: Optional[PhaseSpaceGenerator] = None,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> EventCollection:
    """Facade function for creating (unweighted) phase space samples.

    Args:
        size: Sample size to generate.
        reaction_info: A `ampform.kinematics.ReactionInfo`
            needed for the `.PhaseSpaceGenerator.setup` of the phase space
            generator instanced.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    if phsp_generator is None:
        phsp_generator = TFPhaseSpaceGenerator()
    phsp_generator.setup(reaction_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size,
        desc="Generating phase space sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool = EventCollection({})
    while momentum_pool.n_events < size:
        phsp_sample, weights = phsp_generator.generate(
            bunch_size, random_generator
        )
        hit_and_miss_randoms = random_generator(bunch_size)
        bunch = EventCollection(phsp_sample).select_events(
            weights > hit_and_miss_randoms
        )

        if momentum_pool.n_events > 0:
            momentum_pool.append(bunch)
        else:
            momentum_pool = bunch
        progress_bar.update(n=bunch.n_events)
    finalize_progress_bar(progress_bar)
    return momentum_pool.select_events(slice(0, size))


def finalize_progress_bar(progress_bar: tqdm) -> None:
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()
