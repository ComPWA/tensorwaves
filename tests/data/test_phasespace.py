from pprint import pprint
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest

from tensorwaves.data import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
    TFWeightedPhaseSpaceGenerator,
)
from tensorwaves.interface import DataSample

if TYPE_CHECKING:
    from qrules import ParticleCollection


class TestTFPhaseSpaceGenerator:
    @pytest.mark.parametrize(
        ("initial_state", "final_state", "expected_sample"),
        [
            (
                "J/psi(1S)",
                ("pi0", "pi0", "pi0"),
                {
                    "p0": [
                        [0.841233472, 0.799667989, 0.159823862, 0.156340839],
                        [0.640234742, -0.364360112, -0.371962329, 0.347228344],
                        [0.631540320, 0.403805561, 0.417294074, -0.208401449],
                    ],
                    "p1": [
                        [1.09765205, -0.05378975, -0.53523771, -0.94723204],
                        [1.426564296, 1.168326711, -0.060296302, -0.805136016],
                        [1.243480165, 0.014812643, 0.081738919, 1.233338364],
                    ],
                    "p2": [
                        [1.158014477, -0.745878234, 0.375413844, 0.790891204],
                        [1.030100961, -0.803966599, 0.432258632, 0.457907671],
                        [1.22187951, -0.41861820, -0.49903210, -1.02493691],
                    ],
                },
            ),
            (
                "J/psi(1S)",
                ("pi0", "pi0", "pi0", "gamma"),
                {
                    "p0": [
                        [0.520913076, 0.037458949, 0.339629143, -0.369297399],
                        [1.180624927, -0.569078090, 0.687702756, -0.760836072],
                        [0.606831154, 0.543652274, 0.220242315, -0.077206475],
                    ],
                    "p1": [
                        [0.353305116, 0.130561009, 0.299006221, -0.012444727],
                        [0.194507152, 0.123009165, 0.057692537, 0.033979586],
                        [0.331482507, 0.224048290, -0.156048645, 0.130817046],
                    ],
                    "p2": [
                        [1.276779728, 0.236609937, -0.366594420, 1.192296945],
                        [1.339317905, 0.571746863, -0.586304492, 1.051145223],
                        [0.820720580, 0.402982692, -0.697161285, 0.083274400],
                    ],
                    "p3": [
                        [0.945902080, -0.40462990, -0.27204094, -0.81055482],
                        [0.38245001, -0.12567794, -0.15909080, -0.32428874],
                        [1.337865758, -1.170683257, 0.632967615, -0.136884971],
                    ],
                },
            ),
            (
                "J/psi(1S)",
                ("pi0", "pi0", "pi0", "pi0", "gamma"),
                {
                    "p0": [
                        [1.000150296, 0.715439409, -0.284844373, -0.623772405],
                        [0.353592342, 0.134562969, 0.189723778, 0.229578969],
                        [0.734241552, 0.655088513, -0.205095150, -0.222905673],
                    ],
                    "p1": [
                        [0.537685901, -0.062423993, 0.008278542, -0.516645045],
                        [0.440319420, -0.075102421, -0.215361523, 0.351626927],
                        [0.621720722, -0.569846157, -0.063070826, 0.199036046],
                    ],
                    "p2": [
                        [0.588463958, -0.190428491, -0.002167052, 0.540188288],
                        [0.77747437, -0.11485659, -0.55477746, -0.51505105],
                        [0.543908922, -0.120958419, 0.236101553, -0.455239823],
                    ],
                    "p3": [
                        [0.513251926, -0.286712460, -0.089479316, 0.393698133],
                        [0.593575359, 0.536198573, -0.215753382, -0.007385008],
                        [0.564116725, -0.442948181, -0.261969339, 0.187557768],
                    ],
                    "p4": [
                        [0.457347916, -0.175874464, 0.368212199, 0.206531028],
                        [0.931938511, -0.480802535, 0.796168585, -0.058769834],
                        [0.632912076, 0.478664245, 0.294033763, 0.291551681],
                    ],
                },
            ),
        ],
    )
    def test_generate(
        self,
        initial_state: str,
        final_state: Sequence[str],
        expected_sample: DataSample,
        pdg: "ParticleCollection",
    ):
        sample_size = 3
        rng = TFUniformRealNumberGenerator(seed=0)
        phsp_generator = TFPhaseSpaceGenerator(
            initial_state_mass=pdg[initial_state].mass,
            final_state_masses={
                i: pdg[name].mass for i, name in enumerate(final_state)
            },
        )
        phsp_momenta = phsp_generator.generate(sample_size, rng)
        assert set(phsp_momenta) == set(expected_sample)
        n_events = len(next(iter(expected_sample.values())))
        for i in expected_sample:  # pylint: disable=consider-using-dict-items
            expected_momenta = expected_sample[i]
            momenta = phsp_momenta[i]
            assert len(expected_momenta) == n_events
            assert len(momenta) == n_events
            assert pytest.approx(momenta, abs=1e-6) == expected_sample[i]


class TestTFWeightedPhaseSpaceGenerator:
    def test_generate_deterministic(self, pdg: "ParticleCollection"):
        sample_size = 5
        initial_state_name = "J/psi(1S)"
        final_state_names = ["K0", "Sigma+", "p~"]
        rng = TFUniformRealNumberGenerator(seed=123)
        phsp_generator = TFWeightedPhaseSpaceGenerator(
            initial_state_mass=pdg[initial_state_name].mass,
            final_state_masses={
                i: pdg[name].mass for i, name in enumerate(final_state_names)
            },
        )
        phsp_momenta, weights = phsp_generator.generate(sample_size, rng)
        print("Expected values, get by running pytest with the -s flag")
        pprint(
            {
                i: np.round(four_momenta, decimals=10).tolist()
                for i, four_momenta in phsp_momenta.items()
            }
        )
        expected_sample = {
            "p0": [
                [0.7059154068, 0.3572095625, 0.251997269, 0.2441281612],
                [0.6996310679, -0.3562654953, -0.1367339084, 0.3102348449],
                [0.7592776659, 0.0551489184, 0.3313621005, -0.4648049287],
                [0.7820530714, 0.4694971942, 0.2238765653, -0.3056827887],
                [0.6628957748, 0.1287045232, 0.1927256954, 0.3716262275],
            ],
            "p1": [
                [1.2268366211, 0.0530779071, 0.2808911915, -0.0938614524],
                [1.2983113985, 0.0580707314, -0.345843232, -0.3847489307],
                [1.3730435556, -0.264045346, -0.3231669721, 0.5445096619],
                [1.2694745247, -0.0510249037, -0.3895930085, 0.2063451448],
                [1.3387073694, -0.167841506, -0.5904119798, 0.0279167867],
            ],
            "p2": [
                [1.1641479721, -0.4102874697, -0.5328884605, -0.1502667089],
                [1.0989575336, 0.2981947639, 0.4825771404, 0.0745140857],
                [0.9645787786, 0.2088964277, -0.0081951284, -0.0797047333],
                [1.0453724039, -0.4184722905, 0.1657164432, 0.0993376439],
                [1.0952968558, 0.0391369828, 0.3976862844, -0.3995430142],
            ],
        }
        n_events = len(next(iter(expected_sample.values())))
        assert set(phsp_momenta) == set(expected_sample)
        for i in expected_sample:  # pylint: disable=consider-using-dict-items
            expected_momenta = expected_sample[i]
            momenta = phsp_momenta[i]
        assert len(expected_momenta) == n_events
        assert len(momenta) == n_events
        assert pytest.approx(momenta) == expected_sample[i]

        assert len(weights) == sample_size
        assert pytest.approx(weights) == [
            0.403159552393528,
            0.474671812206617,
            0.267247150423739,
            0.456160360696799,
            0.475480297190723,
        ]
