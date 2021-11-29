# pylint: disable=redefined-outer-name
from pathlib import Path

import pytest
import qrules

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.interface import DataSample
from tensorwaves.model.sympy import SympyModel
from tensorwaves.optimizer.callbacks import (
    CallbackList,
    CSVSummary,
    YAMLSummary,
)
from tensorwaves.optimizer.minuit import Minuit2


@pytest.fixture(scope="session")
def estimator(
    sympy_model: SympyModel, data_set: DataSample, phsp_set: DataSample
) -> UnbinnedNLL:
    return UnbinnedNLL(
        sympy_model,
        dict(data_set),
        dict(phsp_set),
        backend="jax",
    )


def test_fit_and_callbacks(  # pylint: disable=too-many-locals
    estimator: UnbinnedNLL,
    output_dir: Path,
    reaction: qrules.ReactionInfo,
    sympy_model: SympyModel,
):
    formalism_alias = reaction.formalism[:4]
    if sympy_model.max_complexity is None:
        lambdify_type = "normal"
    else:
        lambdify_type = "optimized"
    filename = output_dir / f"fit_result_{formalism_alias}_{lambdify_type}"
    csv_file = f"{filename}.csv"
    yml_file = f"{filename}.yml"
    optimizer = Minuit2(
        callback=CallbackList(
            [
                CSVSummary(csv_file),
                YAMLSummary(yml_file, step_size=1),
            ]
        )
    )

    if reaction.formalism == "canonical-helicity":
        coefficient_name = (
            R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(500) \gamma;"
            R" f_{0}(500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}"
        )
    else:
        coefficient_name = (
            R"C_{J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to"
            R" \pi^{0}_{0} \pi^{0}_{0}}"
        )
    initial_parameters = {
        coefficient_name: 1.0 + 0.0j,
        "Gamma_f(0)(500)": 0.3,
        "m_f(0)(980)": 1,
    }
    fit_result = optimizer.optimize(estimator, initial_parameters)

    final_parameters = fit_result.parameter_values
    csv_parameters = CSVSummary.load_latest_parameters(csv_file)
    yml_parameters = YAMLSummary.load_latest_parameters(yml_file)
    for par, value in final_parameters.items():
        assert pytest.approx(value, rel=1e-2) == csv_parameters[par]
        assert pytest.approx(value, rel=1e-2) == yml_parameters[par]
