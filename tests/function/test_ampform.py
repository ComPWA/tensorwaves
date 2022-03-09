# pylint: disable=import-outside-toplevel
import numpy as np
import pytest

from tensorwaves.data import SympyDataTransformer
from tensorwaves.function._backend import find_function
from tensorwaves.function.sympy import create_parametrized_function


@pytest.mark.parametrize("backend", ["jax", "math", "numba", "numpy", "tf"])
def test_complex_sqrt(backend: str):
    import sympy as sp
    from ampform.sympy.math import ComplexSqrt
    from numpy.lib.scimath import sqrt as complex_sqrt

    x = sp.Symbol("x")
    expr = ComplexSqrt(x)
    function = create_parametrized_function(
        expr.doit(), parameters={}, backend=backend
    )
    if backend == "math":
        values = -4
    else:
        linspace = find_function("linspace", backend)
        kwargs = {}
        if backend == "tf":
            kwargs["dtype"] = find_function("complex64", backend)
        values = linspace(-4, +4, 9, **kwargs)
    data = {"x": values}
    output_array = function(data)  # type: ignore[arg-type]
    np.testing.assert_almost_equal(output_array, complex_sqrt(data["x"]))


@pytest.mark.parametrize("backend", ["jax", "numpy", "tf"])
def test_four_momenta_to_helicity_angles(backend):
    import ampform
    import qrules

    reaction = qrules.generate_transitions(
        initial_state=("J/psi(1S)", [+1]),
        final_state=[("gamma", [+1]), "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(500)"],
        allowed_interaction_types=["EM", "strong"],
    )

    builder = ampform.get_builder(reaction)
    model = builder.formulate()

    expressions = model.kinematic_variables
    converter = SympyDataTransformer.from_sympy(expressions, backend)
    assert set(converter.functions) == {
        "m_0",
        "m_012",
        "m_1",
        "m_12",
        "m_2",
        "phi_0",
        "phi_1^12",
        "theta_0",
        "theta_1^12",
    }

    zeros = np.zeros(shape=(1, 4))
    data_momenta = {"p0": zeros, "p1": zeros, "p2": zeros}
    data = converter(data_momenta)
    for var_name in converter.functions:
        if var_name in {"phi_1^12", "theta_0", "theta_1^12"}:
            assert np.isnan(data[var_name])
        else:
            assert data[var_name] == 0
