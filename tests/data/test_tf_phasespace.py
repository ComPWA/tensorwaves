import numpy as np
import pytest

from tensorwaves.data.tf_phasespace import TFUniformRealNumberGenerator


class TestTFUniformRealNumberGenerator:
    @staticmethod
    def test_deterministic_call():
        generator = TFUniformRealNumberGenerator(seed=123)
        size = 100
        sample = generator(size, min_value=-1, max_value=+1)
        assert len(sample) == size
        assert pytest.approx(float(np.min(sample)), abs=1e-6) == -0.993160
        assert pytest.approx(float(np.max(sample)), abs=1e-6) == 0.999594
        assert pytest.approx(float(np.mean(sample)), abs=1e-6) == -0.027578
