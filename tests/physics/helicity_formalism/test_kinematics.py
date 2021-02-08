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
            "m_2",
            "m_2+3+4",
            "m_3",
            "m_3+4",
            "m_4",
            "phi_3+4_2",
            "phi_3_4_vs_2",
            "theta_3+4_2",
            "theta_3_4_vs_2",
        }
        _, sample_size, _ = data_sample.shape
        assert sample_size == 10000
        final_state = helicity_model.kinematics.final_state
        float_only_variables = {
            "m_2": final_state[2].mass,
            "m_3": final_state[3].mass,
            "m_4": final_state[4].mass,
        }
        for var_name, value in data_set.items():
            if var_name in float_only_variables:
                assert value == float_only_variables[var_name]
            else:
                assert len(value) == sample_size

    def test_convert_to_pandas(self, data_set: dict):
        data_frame = pd.DataFrame(data_set)
        assert set(data_frame.columns) == set(data_frame)
