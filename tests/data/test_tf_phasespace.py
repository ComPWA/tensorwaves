import pytest

from tensorwaves.data.tf_phasespace import TFUniformRealNumberGenerator


class TestTFUniformRealNumberGenerator:
    @staticmethod
    def test_deterministic_call():
        generator = TFUniformRealNumberGenerator(seed=456)
        sample = generator(size=3, min_value=-1, max_value=+1)
        assert pytest.approx(sample) == [-0.38057342, -0.21197986, 0.14724727]
