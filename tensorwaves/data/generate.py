import numpy as np
import logging


def generate_data_bunch(bunch_size, phsp_generator, random_generator,
                        intensity, kinematics):
    phsp_sample, weights = phsp_generator.generate(
        bunch_size, random_generator)
    dataset = kinematics.convert(phsp_sample)
    intensities = intensity(dataset)
    maxvalue = np.max(intensities)

    uniform_randoms = random_generator(bunch_size, max_value=maxvalue)

    phsp_sample = phsp_sample.transpose(1, 0, 2)

    return (phsp_sample[weights * intensities > uniform_randoms], maxvalue)


# @profile
def generate_data(size, phsp_generator, random_generator,
                  intensity, kinematics):
    events = np.array([])

    current_max = 0.0
    bunch_size = 50000

    from progress.bar import Bar
    bar = Bar('Processing', max=size, suffix='%(percent)d%%')

    while np.size(events, 0) < size:
        bunch, maxvalue = generate_data_bunch(bunch_size,
                                              phsp_generator, random_generator,
                                              intensity, kinematics)

        if maxvalue > current_max:
            current_max = 1.05 * maxvalue
            if(np.size(events, 0) > 0):
                logging.info("processed bunch maximum of {} is over current"
                             " maximum {}. Restarting generation!".format(
                                 maxvalue, current_max))
                events = np.array([])
                bar = Bar('Processing', max=size, suffix='%(percent)d%%')
                continue
        if np.size(events) > 0:
            events = np.vstack((events, bunch))
        else:
            events = bunch
        bar.next(np.size(bunch, 0))
    bar.finish()
    return events[0:size].transpose(1, 0, 2)


def generate_phsp(n, phsp_generator, random_generator):
    events = np.array([])

    bunch_size = 50000

    from progress.bar import Bar
    bar = Bar('Processing', max=n, suffix='%(percent)d%%')

    while np.size(events) < n:
        p, w = phsp_generator.generate(bunch_size, random_generator)
        p = p.transpose(1, 0, 2)

        r = random_generator(bunch_size)

        bunch = p[w > r]

        if np.size(events) > 0:
            events = np.vstack((events, bunch))
        else:
            events = bunch
        bar.next(np.size(bunch))
    bar.finish()
    return events[0:n].transpose(1, 0, 2)
