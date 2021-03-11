# pylint: disable=no-self-use

import numpy as np
import pandas as pd
from expertsystem.amplitude.model import AmplitudeModel


class TestHelicityKinematics:
    def test_convert(
        self,
        helicity_model: AmplitudeModel,
        data_sample: np.ndarray,
        data_set: dict,
    ):
        assert set(data_set) == {
            "mSq_0",
            "mSq_0+1+2",
            "mSq_1",
            "mSq_1+2",
            "mSq_2",
            "phi_0_1+2",
            "phi_1_2_vs_0",
            "theta_0_1+2",
            "theta_1_2_vs_0",
        }
        _, sample_size, _ = data_sample.shape
        assert sample_size == 10000
        final_state = helicity_model.kinematics.final_state
        float_only_variables = {
            "mSq_0": final_state[0].mass ** 2,
            "mSq_1": final_state[1].mass ** 2,
            "mSq_2": final_state[2].mass ** 2,
        }
        for var_name, value in data_set.items():
            if var_name in float_only_variables:
                assert value == float_only_variables[var_name]
            else:
                assert len(value) == sample_size

    def test_convert_to_pandas(self, data_set: dict):
        data_frame = pd.DataFrame(data_set)
        assert set(data_frame.columns) == set(data_frame)
