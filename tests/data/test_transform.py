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


def _create_transformer(expressions: dict[sp.Basic, sp.Expr]) -> SympyDataTransformer:
    return SympyDataTransformer.from_sympy(expressions, backend="jax")


def describe_ChainedDataTransformer():
    @pytest.mark.parametrize("extend", [False, True])
    def it_recovers_the_input_when_the_transforms_are_each_other_inverse(extend: bool):
        x, y, v, w = sp.symbols("x y v w")
        transform1 = _create_transformer({v: 2 * x - 5, w: -0.2 * y + 3})
        transform2 = _create_transformer({x: 0.5 * (v + 5), y: 5 * (3 - w)})
        chained_transform = ChainedDataTransformer([transform1, transform2], extend)
        rng = np.random.default_rng(seed=0)
        data = {"x": rng.uniform(size=100), "y": rng.uniform(size=100)}
        transformed_data = chained_transform(data)
        for key in data:  # ruff:ignore[dict-index-missing-items]
            np.testing.assert_allclose(
                data[key],
                transformed_data[key],
                rtol=1e-13,
            )
        if extend:
            assert set(transformed_data) == {"x", "y", "v", "w"}
        else:
            assert set(transformed_data) == {"x", "y"}

    def it_returns_a_new_sample_even_for_a_single_identity_transform():
        transform = IdentityTransformer()
        chained_transform = ChainedDataTransformer([transform])
        data = {
            "x": np.ones(5),
            "y": np.ones(5),
        }
        assert data == chained_transform(data)
        assert data is not chained_transform(data)  # DataSample returned as new dict


def describe_IdentityTransformer():
    def it_returns_the_same_sample_object():
        transform = IdentityTransformer()
        data = {
            "x": np.ones(5),
            "y": np.ones(5),
        }
        assert data is transform(data)


def describe_SympyDataTransformer():
    @pytest.mark.parametrize("backend", ["jax", "numba", "numpy", "tf"])
    def it_converts_polar_to_cartesian_coordinates(backend):
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
        output = converter(input_data)
        assert pytest.approx(output["x"]) == [1, sqrt(2) / 2, 0, -1]
        assert pytest.approx(output["y"]) == [0, sqrt(2) / 2, 1, 0]
