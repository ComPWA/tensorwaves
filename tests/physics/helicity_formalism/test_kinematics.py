# pylint: disable=no-self-use

import numpy as np


class TestHelicityKinematics:
    def test_convert(self, data_sample: np.ndarray, data_set: dict):
        assert set(data_set) == {
            "mSq_2",
            "mSq_2+3+4",
            "mSq_3",
            "mSq_3+4",
            "mSq_4",
            "phi_3+4_2",
            "phi_3_4_vs_2",
            "theta_3+4_2",
            "theta_3_4_vs_2",
        }
        _, sample_size, _ = data_sample.shape
        assert sample_size == 10000
        for array in data_set.values():
            assert len(array) == sample_size
