from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest

from tensorwaves.data import generate_data, generate_phsp
from tensorwaves.data.phasespace import TFUniformRealNumberGenerator
from tensorwaves.data.transform import IdentityTransformer
from tensorwaves.interface import DataSample, Function

if TYPE_CHECKING:
    from qrules import ParticleCollection


class FlatDistribution(Function[DataSample, np.ndarray]):
    def __call__(self, data: DataSample) -> np.ndarray:
        some_key = next(iter(data))
        sample_size = len(data[some_key])
        return np.ones(sample_size)


def test_generate_data():
    sample_size = 5
    initial_state_mass = 3.0
    final_state_masses = {0: 0.135, 1: 0.135, 2: 0.135}
    phsp = generate_phsp(
        sample_size,
        initial_state_mass,
        final_state_masses,
        random_generator=TFUniformRealNumberGenerator(seed=0),
    )

    data = generate_data(
        sample_size,
        initial_state_mass,
        final_state_masses,
        data_transformer=IdentityTransformer(),
        intensity=FlatDistribution(),
        random_generator=TFUniformRealNumberGenerator(seed=0),
    )
    assert set(phsp) == {f"p{i}" for i in final_state_masses}
    assert set(phsp) == set(data)
    for i in phsp:
        assert pytest.approx(phsp[i]) == data[i]


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
def test_generate_phsp(
    initial_state: str,
    final_state: Sequence[str],
    expected_sample: DataSample,
    pdg: "ParticleCollection",
):
    sample_size = 3
    rng = TFUniformRealNumberGenerator(seed=0)
    phsp_momenta = generate_phsp(
        sample_size,
        initial_state_mass=pdg[initial_state].mass,
        final_state_masses={
            i: pdg[name].mass for i, name in enumerate(final_state)
        },
        random_generator=rng,
    )
    assert set(phsp_momenta) == set(expected_sample)
    n_events = len(next(iter(expected_sample.values())))
    for i in expected_sample:  # pylint: disable=consider-using-dict-items
        expected_momenta = expected_sample[i]
        momenta = phsp_momenta[i]
        assert len(expected_momenta) == n_events
        assert len(momenta) == n_events
        assert pytest.approx(momenta, abs=1e-6) == expected_sample[i]
