import pytest
import tensorflow_probability as tfp
from expertsystem.amplitude.model import AmplitudeModel

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.data.tf_phasespace import TFUniformRealNumberGenerator
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)


def test_generate_data(helicity_model: AmplitudeModel):
    model = helicity_model
    kinematics = HelicityKinematics.from_model(model)
    seed_stream = tfp.util.SeedStream(seed=0, salt="")
    rng_phsp = TFUniformRealNumberGenerator(seed=seed_stream())
    phsp_sample = generate_phsp(100, kinematics, random_generator=rng_phsp)
    builder = IntensityBuilder(model.particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(model)
    sample_size = 3
    rng_data = TFUniformRealNumberGenerator(seed=seed_stream())
    sample = generate_data(
        sample_size, kinematics, intensity, random_generator=rng_data
    )
    assert pytest.approx(sample, abs=1e-8) == [
        [
            [-0.62925318, 0.00699376, -1.23769616, 1.38848848],
            [0.97025676, 0.40811957, -0.87950715, 1.37167511],
            [-0.67687867, -0.02131860, -1.21579784, 1.39168373],
        ],
        [
            [0.03314958, 0.07888740, -0.13122441, 0.20678660],
            [-1.06531265, -0.44630718, 0.98639142, 1.52488292],
            [0.68586622, 0.04871419, 1.34472782, 1.51634336],
        ],
        [
            [0.59610360, -0.08588116, 1.36892057, 1.50162490],
            [0.09505589, 0.03818760, -0.10688427, 0.20034195],
            [-0.00898755, -0.02739558, -0.12892997, 0.18887289],
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
                    [0.37588781, 0.24997872, -0.23111435, 0.52479861],
                    [-0.11211889, -0.21799969, -0.00730057, 0.27994025],
                    [0.56445616, 0.48745751, -0.19376371, 0.78229707],
                ],
                [
                    [-0.69356482, 0.66277682, -0.72185047, 1.20813580],
                    [0.50329618, 0.70817500, -1.17756203, 1.46958836],
                    [0.41244154, 0.66191549, 0.04316059, 0.79266747],
                ],
                [
                    [0.31767701, -0.91275554, 0.95296483, 1.36396557],
                    [-0.39117729, -0.49017531, 1.18486260, 1.34737138],
                    [-0.97689771, -1.14937301, 0.15060312, 1.52193544],
                ],
            ],
        ),
        (
            ("J/psi(1S)"),
            ("pi0", "pi0", "pi0", "gamma"),
            [
                [
                    [0.01435857, 0.31784899, 0.68053252, 0.76326758],
                    [-0.28665654, 0.19376015, 0.40777581, 0.55155672],
                    [0.78279918, 0.21918933, 0.96086673, 1.26582073],
                ],
                [
                    [-0.32318671, -0.18814549, -0.26905723, 0.48006136],
                    [0.58230393, -0.32748380, -0.05276625, 0.68361284],
                    [0.25065665, 0.06356561, -0.31903259, 0.43228449],
                ],
                [
                    [-0.22189595, 0.37739191, -0.87341635, 0.98627447],
                    [-0.47109108, 0.36601058, -0.98971066, 1.16345884],
                    [-0.76607487, -0.20730812, -0.75038481, 1.10051964],
                ],
                [
                    [0.53072408, -0.50709540, 0.46194106, 0.86729658],
                    [0.17544369, -0.23228692, 0.63470110, 0.69827157],
                    [-0.26738096, -0.07544682, 0.10855068, 0.29827512],
                ],
            ],
        ),
        (
            "J/psi(1S)",
            ("pi0", "pi0", "pi0", "pi0", "gamma"),
            [
                [
                    [-0.23766071, -0.79628480, 0.09781076, 0.84754809],
                    [-0.38287075, -0.83027647, -0.70194699, 1.16055907],
                    [-0.01217064, 0.02832812, 0.33053409, 0.35836033],
                ],
                [
                    [0.26794913, 0.01347832, 0.36551906, 0.47307644],
                    [0.05790591, 0.47110508, 0.73956012, 0.88907874],
                    [-0.22815261, 0.52442322, 0.42401920, 0.72462704],
                ],
                [
                    [0.01320446, -0.20361375, -0.14174539, 0.28274266],
                    [-0.06212328, -0.06824339, -0.11123575, 0.19775892],
                    [0.46210230, 0.40072547, -0.15713046, 0.64577717],
                ],
                [
                    [-0.34024771, 0.66346421, -0.60645487, 0.97054599],
                    [0.19259851, 0.50583191, 0.21364765, 0.59734761],
                    [0.04678297, -0.13690816, 0.13441109, 0.23920194],
                ],
                [
                    [0.29675483, 0.32295601, 0.28487044, 0.52298679],
                    [0.19448960, -0.07841713, -0.14002503, 0.25215563],
                    [-0.26856201, -0.81656865, -0.73183393, 1.12893349],
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
