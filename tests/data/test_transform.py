from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from numpy import sqrt

from tensorwaves.data.transform import (
    ChainedDataTransformer,
    IdentityTransformer,
    SympyDataTransformer,
)


class TestChainedDataTransformer:
    @pytest.mark.parametrize("extend", [False, True])
    def test_identity_chain(self, extend: bool):
        x, y, v, w = sp.symbols("x y v w")
        transform1 = _create_transformer({v: 2 * x - 5, w: -0.2 * y + 3})
        transform2 = _create_transformer({x: 0.5 * (v + 5), y: 5 * (3 - w)})
        chained_transform = ChainedDataTransformer([transform1, transform2], extend)
        rng = np.random.default_rng(seed=0)
        data = {"x": rng.uniform(size=100), "y": rng.uniform(size=100)}
        transformed_data = chained_transform(data)
        for key in data:  # noqa: PLC0206
            np.testing.assert_allclose(
                data[key],
                transformed_data[key],
                rtol=1e-13,
            )
        if extend:
            assert set(transformed_data) == {"x", "y", "v", "w"}
        else:
            assert set(transformed_data) == {"x", "y"}

    def test_single_chain(self):
        transform = IdentityTransformer()
        chained_transform = ChainedDataTransformer([transform])
        data = {
            "x": np.ones(5),
            "y": np.ones(5),
        }
        assert data == chained_transform(data)
        assert data is not chained_transform(data)  # DataSample returned as new dict


def _create_transformer(expressions: dict[sp.Symbol, sp.Expr]) -> SympyDataTransformer:
    return SympyDataTransformer.from_sympy(expressions, backend="jax")


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
