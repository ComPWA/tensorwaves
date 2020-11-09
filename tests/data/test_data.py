# pylint: disable=no-self-use

import numpy as np
import pandas as pd
import pytest
from expertsystem.amplitude.model import AmplitudeModel

from tensorwaves.data._data_frame import MOMENTUM_LABELS, WEIGHT_LABEL
from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
)


class TestPwaAccessor:
    @staticmethod
    def test_properties(helicity_model: AmplitudeModel):
        model = helicity_model
        kinematics = HelicityKinematics.from_model(model)
        n_events = 100
        sample = generate_phsp(n_events, kinematics)
        assert pytest.approx(sample.pwa.mass.mean().to_list(), abs=1e-4) == [
            0,
            0.135,
            0.135,
        ]
        assert sample.pwa.particles == [2, 3, 4]
        assert sample.pwa.particles == sample.pwa.column_names
        assert sample.pwa.weights is None
        assert sample.pwa.intensities is None

        weights = np.ones(n_events)
        sample[WEIGHT_LABEL] = weights
        assert (sample.pwa.weights == weights).all()
        assert (sample.pwa.intensities == weights).all()

        total_p4 = sample.pwa.p4sum
        assert pytest.approx(total_p4.pwa.mass.mean(), abs=1e-6) == 3.0969

        gamma_sample = sample[2]
        assert gamma_sample.pwa.particles is None
        assert gamma_sample.pwa.column_names == MOMENTUM_LABELS
        assert pytest.approx(gamma_sample.pwa.mass.mean(), abs=1e-4) == 0
        assert gamma_sample.equals(gamma_sample.pwa.p4sum)

    @pytest.mark.parametrize(
        "frame",
        [
            pd.DataFrame(
                data=np.ones((3, 24)),
                columns=pd.MultiIndex.from_tuples(
                    [
                        (col1, col2, mom)
                        for col1 in ["col1_A", "col1_B"]
                        for col2 in ["col2_A", "col2_B", "col2_C"]
                        for mom in MOMENTUM_LABELS
                    ]
                ),
            ),
            pd.DataFrame(
                data=np.ones((3, 9)),
                columns=pd.MultiIndex.from_tuples(
                    [
                        (state_id, mom)
                        for state_id in [2, 3, 4]
                        for mom in MOMENTUM_LABELS[:3]
                    ]
                ),
            ),
            pd.DataFrame(data=np.ones((3, 2)), columns=["one", "two"]),
        ],
    )
    def test_validator(self, frame: pd.DataFrame):
        with pytest.raises(IOError):
            assert frame.pwa.mass
