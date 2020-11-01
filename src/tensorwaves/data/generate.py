"""Tools to facilitate data sample generation."""

import logging
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from tensorwaves.data.tf_phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.interfaces import (
    Function,
    Kinematics,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)


def _generate_data_bunch(
    bunch_size: int,
    phsp_generator: PhaseSpaceGenerator,
    random_generator: UniformRealNumberGenerator,
    intensity: Function,
    kinematics: Kinematics,
) -> np.ndarray:
    phsp_sample, weights = phsp_generator.generate(
        bunch_size, random_generator
    )
    dataset = kinematics.convert(phsp_sample)
    intensities = intensity(dataset)
    maxvalue = np.max(intensities)

    uniform_randoms = random_generator(bunch_size, max_value=maxvalue)

    phsp_sample = phsp_sample.transpose(1, 0, 2)

    return (phsp_sample[weights * intensities > uniform_randoms], maxvalue)


def generate_data(
    size: int,
    kinematics: HelicityKinematics,
    intensity: Function,
    phsp_generator: Callable[
        [ParticleReactionKinematicsInfo], PhaseSpaceGenerator
    ] = TFPhaseSpaceGenerator,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> np.ndarray:
    """Facade function for creating data samples based on an intensities.

    Args:
        size: Sample size to generate.
        kinematics: A kinematics instance. Note that this instance must have a
            property :attr:`~.HelicityKinematics.reaction_kinematics_info` of
            the type `.ParticleReactionKinematicsInfo`, otherwise the phase
            space generator instance cannot be constructed.
        intensity: The intensity which will be sampled.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with a default seed of
            :code:`123456`.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    phsp_gen_instance = phsp_generator(kinematics.reaction_kinematics_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator(123456)

    progress_bar = tqdm(
        total=size / bunch_size,
        desc="Generating intensity-based sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    # IncrementalBar(
    #     "Generating", max=size, suffix="%(percent)d%% - %(elapsed_td)s"
    # )
    events = np.array([])
    current_max = 0.0
    while np.size(events, 0) < size:
        bunch, maxvalue = _generate_data_bunch(
            bunch_size,
            phsp_gen_instance,
            random_generator,
            intensity,
            kinematics,
        )

        if maxvalue > current_max:
            current_max = 1.05 * maxvalue
            if np.size(events, 0) > 0:
                logging.info(
                    "processed bunch maximum of %s is over current"
                    " maximum %s. Restarting generation!",
                    maxvalue,
                    current_max,
                )
                events = np.array([])
                progress_bar.update()
                continue
        if np.size(events, 0) > 0:
            events = np.vstack((events, bunch))
        else:
            events = bunch
        progress_bar.update()
    progress_bar.close()
    return events[0:size].transpose(1, 0, 2)


def generate_phsp(
    size: int,
    kinematics: HelicityKinematics,
    phsp_generator: Callable[
        [ParticleReactionKinematicsInfo], PhaseSpaceGenerator
    ] = TFPhaseSpaceGenerator,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> np.ndarray:
    """Facade function for creating (unweighted) phase space samples.

    Args:
        size: Sample size to generate.
        kinematics: A kinematics instance. Note that this instance must have a
            property :attr:`~.HelicityKinematics.reaction_kinematics_info` of
            the type `.ParticleReactionKinematicsInfo`, otherwise the phase
            space generator instance cannot be constructed.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with a default seed of
            :code:`123456`.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    phsp_gen_instance = phsp_generator(kinematics.reaction_kinematics_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator(123456)

    progress_bar = tqdm(
        total=size / bunch_size,
        desc="Generating phase space sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    events = np.array([])
    while np.size(events, 0) < size:
        four_momenta, weights = phsp_gen_instance.generate(
            bunch_size, random_generator
        )
        four_momenta = four_momenta.transpose(1, 0, 2)

        hit_and_miss_randoms = random_generator(bunch_size)

        bunch = four_momenta[weights > hit_and_miss_randoms]

        if np.size(events, 0) > 0:
            events = np.vstack((events, bunch))
        else:
            events = bunch
        progress_bar.update()
    progress_bar.close()
    return events[0:size].transpose(1, 0, 2)
