"""Tools to facilitate data sample generation."""

import logging

import numpy as np

from progress.bar import Bar

from tensorwaves.interfaces import (
    Function,
    Kinematics,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)

from tensorwaves.data.tf_phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
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
    intensity: Function,
    kinematics: Kinematics,
    phsp_generator: PhaseSpaceGenerator = TFPhaseSpaceGenerator,
    random_generator: UniformRealNumberGenerator = TFUniformRealNumberGenerator,
    seed: int = 123456,
    bunch_size: int = 50000,
) -> np.ndarray:
    """Create a data sample based on an intensity."""
    events = np.array([])

    phsp_generator = phsp_generator(
        kinematics.initial_state_mass, kinematics.final_state_masses,
    )

    random_generator = random_generator(seed)

    current_max = 0.0

    progress_bar = Bar("Generating", max=size, suffix="%(percent)d%%")

    while np.size(events, 0) < size:
        bunch, maxvalue = _generate_data_bunch(
            bunch_size, phsp_generator, random_generator, intensity, kinematics
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
                progress_bar = Bar(
                    "Generating", max=size, suffix="%(percent)d%%"
                )
                continue
        if np.size(events, 0) > 0:
            events = np.vstack((events, bunch))
        else:
            events = bunch
        progress_bar.next(np.size(bunch, 0))
    progress_bar.finish()
    return events[0:size].transpose(1, 0, 2)


def generate_phsp(
    size: int,
    kinematics: Kinematics,
    phsp_generator: PhaseSpaceGenerator = TFPhaseSpaceGenerator,
    random_generator: UniformRealNumberGenerator = TFUniformRealNumberGenerator,
    seed: int = 123456,
    bunch_size: int = 50000,
) -> np.ndarray:
    """Facade function for creating phase space samples."""
    events = np.array([])

    phsp_generator = phsp_generator(
        kinematics.initial_state_mass, kinematics.final_state_masses,
    )

    random_generator = random_generator(seed)

    progress_bar = Bar("Generating", max=size, suffix="%(percent)d%%")

    while np.size(events, 0) < size:
        four_momenta, weights = phsp_generator.generate(
            bunch_size, random_generator
        )
        four_momenta = four_momenta.transpose(1, 0, 2)

        hit_and_miss_randoms = random_generator(bunch_size)

        bunch = four_momenta[weights > hit_and_miss_randoms]

        if np.size(events, 0) > 0:
            events = np.vstack((events, bunch))
        else:
            events = bunch
        progress_bar.next(np.size(bunch, 0))
    progress_bar.finish()
    return events[0:size].transpose(1, 0, 2)
