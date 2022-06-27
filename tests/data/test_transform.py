# pylint: disable=invalid-name
import numpy as np
import pytest
import sympy as sp
from numpy import sqrt

from tensorwaves.data import IdentityTransformer, SympyDataTransformer


class TestIdentityTransformer:
    def test_call(self):
        transform = IdentityTransformer()
        data = {
            "x": np.ones(5),
            "y": np.ones(5),
        }
        assert data is transform(data)


class TestSympyDataTransformer:
    @pytest.mark.parametrize("backend", ["jax", "numba", "numpy", "tf"])
    def test_polar_to_cartesian_coordinates(self, backend):
        r, phi, x, y = sp.symbols("r phi x y")
        expressions = {
            x: r * sp.cos(phi),
            y: r * sp.sin(phi),
        }
        converter = SympyDataTransformer.from_sympy(expressions, backend)
        assert set(converter.functions) == {"x", "y"}
        input_data = {
            "r": np.ones(4),
            "phi": np.array([0, np.pi / 4, np.pi / 2, np.pi]),
        }
        output = converter(input_data)  # type: ignore[arg-type]
        assert pytest.approx(output["x"]) == [1, sqrt(2) / 2, 0, -1]
        assert pytest.approx(output["y"]) == [0, sqrt(2) / 2, 1, 0]
