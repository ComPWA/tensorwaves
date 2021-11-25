# cspell:ignore tolist
from pprint import pprint

import numpy as np
import pytest

from tensorwaves.interface import DataSample


def test_generate_data(data_sample: DataSample):
    sample_size = 5
    sub_sample = {k: v[:sample_size] for k, v in data_sample.items()}
    print("Expected values, get by running pytest with the -s flag")
    pprint(
        {
            i: np.round(four_momenta, decimals=11).tolist()
            for i, four_momenta in sub_sample.items()
        }
    )
    expected_sample = {
        0: [
            [1.50757377596, 0.37918944935, 0.73396599969, 1.26106620078],
            [1.41389525301, -0.07315064441, -0.21998573758, 1.39475985207],
            [1.52128570461, 0.06569896528, -1.51812710851, 0.0726906006],
            [1.51480310845, 1.40672331053, 0.49678572189, -0.26260603856],
            [1.52384281483, 0.79694939592, 1.29832389761, -0.03638188481],
        ],
        1: [
            [1.42066087326, -0.34871369761, -0.72119471428, -1.1654765212],
            [0.96610319301, -0.26739932067, -0.15455480956, -0.90539883872],
            [0.60647770024, 0.11616448713, 0.57584161239, -0.06714695611],
            [1.01045883083, -0.88651015826, -0.46024226278, 0.0713099651],
            [1.04324742713, -0.48051670276, -0.91259832182, -0.08009031815],
        ],
        2: [
            [0.16866535079, -0.03047575173, -0.01277128542, -0.09558967958],
            [0.71690155399, 0.34054996508, 0.37454054715, -0.48936101336],
            [0.96913659515, -0.18186345241, 0.94228549612, -0.00554364449],
            [0.57163806072, -0.52021315227, -0.03654345912, 0.19129607347],
            [0.52980975805, -0.31643269316, -0.38572557579, 0.11647220296],
        ],
    }
    n_events = len(next(iter(expected_sample.values())))
    assert set(sub_sample) == set(expected_sample)
    for i in expected_sample:  # pylint: disable=consider-using-dict-items
        expected_momenta = expected_sample[i]
        momenta = sub_sample[i]
        assert len(expected_momenta) == n_events
        assert len(momenta) == n_events
        assert pytest.approx(momenta) == expected_sample[i]
