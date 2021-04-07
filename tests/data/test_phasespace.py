from pprint import pprint

import numpy as np
import pytest
from ampform.kinematics import ReactionInfo

from tensorwaves.data.phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)


class TestTFPhaseSpaceGenerator:
    @staticmethod
    def test_generate_deterministic(pdg):
        sample_size = 5
        initial_state_name = "J/psi(1S)"
        final_state_names = ["K0", "Sigma+", "p~"]
        reaction_info = ReactionInfo(
            initial_state={-1: pdg[initial_state_name]},
            final_state={
                i: pdg[name] for i, name in enumerate(final_state_names)
            },
        )
        rng = TFUniformRealNumberGenerator(seed=123)
        phsp_generator = TFPhaseSpaceGenerator()
        phsp_generator.setup(reaction_info)
        momentum_pool, weights = phsp_generator.generate(sample_size, rng)
        print("Expected values, get by running pytest with the -s flag")
        pprint(
            {
                i: np.round(four_momenta, decimals=10).tolist()
                for i, four_momenta in momentum_pool.items()
            }
        )
        expected_sample = {
            0: [
                [0.7059154068, 0.3572095625, 0.251997269, 0.2441281612],
                [0.6996310679, -0.3562654953, -0.1367339084, 0.3102348449],
                [0.7592776659, 0.0551489184, 0.3313621005, -0.4648049287],
                [0.7820530714, 0.4694971942, 0.2238765653, -0.3056827887],
                [0.6628957748, 0.1287045232, 0.1927256954, 0.3716262275],
            ],
            1: [
                [1.2268366211, 0.0530779071, 0.2808911915, -0.0938614524],
                [1.2983113985, 0.0580707314, -0.345843232, -0.3847489307],
                [1.3730435556, -0.264045346, -0.3231669721, 0.5445096619],
                [1.2694745247, -0.0510249037, -0.3895930085, 0.2063451448],
                [1.3387073694, -0.167841506, -0.5904119798, 0.0279167867],
            ],
            2: [
                [1.1641479721, -0.4102874697, -0.5328884605, -0.1502667089],
                [1.0989575336, 0.2981947639, 0.4825771404, 0.0745140857],
                [0.9645787786, 0.2088964277, -0.0081951284, -0.0797047333],
                [1.0453724039, -0.4184722905, 0.1657164432, 0.0993376439],
                [1.0952968558, 0.0391369828, 0.3976862844, -0.3995430142],
            ],
        }
        assert set(momentum_pool) == set(expected_sample)
        for i, momenta in momentum_pool.items():
            assert len(momenta) == len(expected_sample[i])
            assert pytest.approx(momenta) == expected_sample[i]

        assert len(weights) == sample_size
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
