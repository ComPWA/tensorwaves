import pytest
from expertsystem.amplitude.kinematics import ReactionInfo

from tensorwaves.data.tf_phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)


class TestTFPhaseSpaceGenerator:
    @staticmethod
    def test_generate_deterministic(pdg):
        sample_size = 5
        initial_state_names = ["J/psi(1S)"]
        final_state_names = ["K0", "Sigma+", "p~"]
        reaction_info = ReactionInfo(
            initial_state={
                i: pdg[name]
                for i, name in zip(
                    range(-len(initial_state_names) - 1, 0),
                    initial_state_names,
                )
            },
            final_state={
                i: pdg[name]
                for i, name in zip(
                    range(-len(final_state_names) - 1, 0), final_state_names
                )
            },
        )
        rng = TFUniformRealNumberGenerator(seed=123)
        phsp_generator = TFPhaseSpaceGenerator(reaction_info)
        four_momenta, weights = phsp_generator.generate(sample_size, rng)
        for values in four_momenta.values():
            assert values.shape == (sample_size, 4)
        assert weights.shape == (sample_size,)
        assert pytest.approx(four_momenta, abs=1e-6) == [
            [
                [0.357209, 0.251997, 0.244128, 0.705915],
                [-0.356265, -0.136733, 0.310234, 0.699631],
                [0.055148, 0.331362, -0.464804, 0.759277],
                [0.469497, 0.223876, -0.305682, 0.782053],
                [0.128704, 0.192725, 0.371626, 0.662895],
            ],
            [
                [0.053077, 0.280891, -0.093861, 1.226836],
                [0.058070, -0.345843, -0.384748, 1.298311],
                [-0.264045, -0.323166, 0.544509, 1.373043],
                [-0.051024, -0.389593, 0.206345, 1.269474],
                [-0.167841, -0.590411, 0.027916, 1.338707],
            ],
            [
                [-0.410287, -0.532888, -0.150266, 1.164147],
                [0.298194, 0.482577, 0.074514, 1.098957],
                [0.208896, -0.008195, -0.079704, 0.964578],
                [-0.418472, 0.165716, 0.099337, 1.045372],
                [0.039136, 0.397686, -0.399543, 1.095296],
            ],
        ]
        assert pytest.approx(weights) == [
            0.403159552393528,
            0.474671812206617,
            0.267247150423739,
            0.456160360696799,
            0.475480297190723,
        ]


class TestTFUniformRealNumberGenerator:
    @staticmethod
    def test_deterministic_call():
        generator = TFUniformRealNumberGenerator(seed=456)
        sample = generator(size=3, min_value=-1, max_value=+1)
        assert pytest.approx(sample) == [-0.38057342, -0.21197986, 0.14724727]
