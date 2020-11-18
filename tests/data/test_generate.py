import numpy as np
import pytest

from tensorwaves.data.generate import generate_phsp
from tensorwaves.data.tf_phasespace import TFUniformRealNumberGenerator
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)


def test_generate_data(data_sample: np.ndarray):
    sub_sample = data_sample[:, :5, :]
    assert pytest.approx(sub_sample) == [
        [
            [-0.4285702231, 0.1489909537, 1.3762794555, 1.4491431537],
            [0.4685107678, -0.4318226308, -1.3416885532, 1.4852950205],
            [-0.1531690632, 0.7138889754, -1.1701068519, 1.3792201695],
            [-0.8517097194, 0.1734224426, 0.3051183486, 0.9211851043],
            [0.5186454254, -0.5183265906, -0.7991097798, 1.0845422869],
        ],
        [
            [0.3699375689, 0.2315795106, -0.4782764324, 0.6613999978],
            [-0.5281639952, 0.2582456887, 1.2006268330, 1.3436411612],
            [0.2355328204, -0.9348966393, 0.8092391521, 1.2659360875],
            [1.3161570082, 0.3132105485, -0.5216295386, 1.4562576111],
            [0.0861292048, 1.0265051732, 0.5349731779, 1.1685658510],
        ],
        [
            [0.0586326541, -0.3805704643, -0.8980030230, 0.9863568484],
            [0.0596532273, 0.1735769421, 0.1410617202, 0.2679638181],
            [-0.0823637571, 0.2210076639, 0.3608676997, 0.4517437429],
            [-0.4644472887, -0.4866329911, 0.2165111900, 0.7194572844],
            [-0.6047746303, -0.5081785825, 0.2641366019, 0.8437918619],
        ],
    ]


@pytest.mark.parametrize(
    "initial_state_names, final_state_names, expected_sample",
    [
        (
            "J/psi(1S)",
            ("pi0", "pi0", "pi0"),
            [
                [
                    [0.799667989, 0.159823862, 0.156340839, 0.841233472],
                    [-0.364360112, -0.371962329, 0.347228344, 0.640234742],
                    [0.403805561, 0.417294074, -0.208401449, 0.631540320],
                ],
                [
                    [-0.053789754, -0.535237707, -0.947232044, 1.097652050],
                    [1.168326711, -0.060296302, -0.805136016, 1.426564296],
                    [0.014812643, 0.081738919, 1.233338364, 1.243480165],
                ],
                [
                    [-0.745878234, 0.375413844, 0.790891204, 1.158014477],
                    [-0.803966599, 0.432258632, 0.457907671, 1.030100961],
                    [-0.418618204, -0.499032994, -1.024936914, 1.221879513],
                ],
            ],
        ),
        (
            ("J/psi(1S)"),
            ("pi0", "pi0", "pi0", "gamma"),
            [
                [
                    [0.037458949, 0.339629143, -0.369297399, 0.520913076],
                    [-0.569078090, 0.687702756, -0.760836072, 1.180624927],
                    [0.543652274, 0.220242315, -0.077206475, 0.606831154],
                ],
                [
                    [0.130561009, 0.299006221, -0.012444727, 0.353305116],
                    [0.123009165, 0.057692537, 0.033979586, 0.194507152],
                    [0.224048290, -0.156048645, 0.130817046, 0.331482507],
                ],
                [
                    [0.236609937, -0.366594420, 1.192296945, 1.276779728],
                    [0.571746863, -0.586304492, 1.051145223, 1.339317905],
                    [0.402982692, -0.697161285, 0.083274400, 0.820720580],
                ],
                [
                    [-0.404629896, -0.272040943, -0.810554818, 0.945902078],
                    [-0.125677938, -0.159090801, -0.324288738, 0.382450013],
                    [-1.170683257, 0.632967615, -0.136884971, 1.337865758],
                ],
            ],
        ),
        (
            "J/psi(1S)",
            ("pi0", "pi0", "pi0", "pi0", "gamma"),
            [
                [
                    [0.715439409, -0.284844373, -0.623772405, 1.000150296],
                    [0.134562969, 0.189723778, 0.229578969, 0.353592342],
                    [0.655088513, -0.205095150, -0.222905673, 0.734241552],
                ],
                [
                    [-0.062423993, 0.008278542, -0.516645045, 0.537685901],
                    [-0.075102421, -0.215361523, 0.351626927, 0.440319420],
                    [-0.569846157, -0.063070826, 0.199036046, 0.621720722],
                ],
                [
                    [-0.190428491, -0.002167052, 0.540188288, 0.588463958],
                    [-0.114856586, -0.554777459, -0.515051054, 0.777474366],
                    [-0.120958419, 0.236101553, -0.455239823, 0.543908922],
                ],
                [
                    [-0.286712460, -0.089479316, 0.393698133, 0.513251926],
                    [0.536198573, -0.215753382, -0.007385008, 0.593575359],
                    [-0.442948181, -0.261969339, 0.187557768, 0.564116725],
                ],
                [
                    [-0.175874464, 0.368212199, 0.206531028, 0.457347916],
                    [-0.480802535, 0.796168585, -0.058769834, 0.931938511],
                    [0.478664245, 0.294033763, 0.291551681, 0.632912076],
                ],
            ],
        ),
    ],
)
def test_generate_phsp(
    initial_state_names, final_state_names, expected_sample, pdg
):
    reaction_info = ParticleReactionKinematicsInfo(
        initial_state_names, final_state_names, pdg
    )
    kin = HelicityKinematics(reaction_info)
    sample_size = 3
    rng = TFUniformRealNumberGenerator(seed=0)
    sample = generate_phsp(sample_size, kin, random_generator=rng)
    assert sample.shape == (len(final_state_names), sample_size, 4)
    assert pytest.approx(sample, abs=1e-8) == expected_sample
