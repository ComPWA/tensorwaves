"""Tools to facilitate data sample generation."""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from expertsystem.amplitude.data import MomentumPool
from expertsystem.amplitude.kinematics import HelicityKinematics, ReactionInfo
from tqdm import tqdm

from tensorwaves.data.tf_phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.interfaces import (
    Function,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)


def _generate_data_bunch(
    bunch_size: int,
    phsp_generator: PhaseSpaceGenerator,
    random_generator: UniformRealNumberGenerator,
    intensity: Function,
    kinematics: HelicityKinematics,
) -> Tuple[MomentumPool, float]:
    phsp_sample, weights = phsp_generator.generate(
        bunch_size, random_generator
    )
    momentum_pool = MomentumPool(phsp_sample)
    dataset = kinematics.convert(momentum_pool)
    intensities = intensity(dataset)
    maxvalue: float = np.max(intensities)

    uniform_randoms = random_generator(bunch_size, max_value=maxvalue)

    hit_and_miss_sample = momentum_pool.select_events(
        weights * intensities > uniform_randoms
    )
    return hit_and_miss_sample, maxvalue


def generate_data(
    size: int,
    kinematics: HelicityKinematics,
    intensity: Function,
    phsp_generator: Callable[
        [ReactionInfo], PhaseSpaceGenerator
    ] = TFPhaseSpaceGenerator,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> MomentumPool:
    """Facade function for creating data samples based on an intensities.

    Args:
        size: Sample size to generate.
        kinematics: A `~expertsystem.amplitude.kinematics.HelicityKinematics`
            instance.
        intensity: The intensity `.Function` that will be sampled.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    phsp_gen_instance = phsp_generator(kinematics.reaction_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size / bunch_size,
        desc="Generating intensity-based sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool = MomentumPool({})
    current_max = 0.0
    while momentum_pool.n_events < size:
        bunch, maxvalue = _generate_data_bunch(
            bunch_size,
            phsp_gen_instance,
            random_generator,
            intensity,
            kinematics,
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
                momentum_pool = MomentumPool({})
                progress_bar.update()
                continue
        if np.size(momentum_pool, 0) > 0:
            momentum_pool.append(bunch)
        else:
            momentum_pool = bunch
        progress_bar.update()
    progress_bar.close()
    return momentum_pool.select_events(slice(0, size))


def generate_phsp(
    size: int,
    kinematics: HelicityKinematics,
    phsp_generator: Callable[
        [ReactionInfo], PhaseSpaceGenerator
    ] = TFPhaseSpaceGenerator,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> MomentumPool:
    """Facade function for creating (unweighted) phase space samples.

    Args:
        size: Sample size to generate.
        kinematics: A kinematics instance. Note that this instance must have a
            property
            `~expertsystem.amplitude.kinematics.HelicityKinematics.reaction_info`
            of the type
            `expertsystem.amplitude.kinematics.ReactionInfo`,
            otherwise the phase space generator instance cannot be constructed.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    phsp_gen_instance = phsp_generator(kinematics.reaction_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size / bunch_size,
        desc="Generating phase space sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool = MomentumPool({})
    while momentum_pool.n_events < size:
        phsp_sample, weights = phsp_gen_instance.generate(
            bunch_size, random_generator
        )
        hit_and_miss_randoms = random_generator(bunch_size)
        bunch = MomentumPool(phsp_sample).select_events(
            weights > hit_and_miss_randoms
        )

        if momentum_pool.n_events > 0:
            momentum_pool.append(bunch)
        else:
            momentum_pool = bunch
        progress_bar.update()
    progress_bar.close()
    return momentum_pool.select_events(slice(0, size))
