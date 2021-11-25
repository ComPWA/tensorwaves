from pathlib import Path
from typing import Type

import pytest
import qrules

from tensorwaves.interface import FitResult
from tensorwaves.model.sympy import SympyModel
from tensorwaves.optimizer.callbacks import CSVSummary, Loadable, YAMLSummary


@pytest.mark.parametrize(
    ("callback_type", "filename"),
    [
        (CSVSummary, "fit_traceback_{}_{}.csv"),
        (YAMLSummary, "fit_result_{}_{}.yml"),
    ],
)
def test_load_latest_parameters(
    reaction: qrules.ReactionInfo,
    sympy_model: SympyModel,
    callback_type: Type[Loadable],
    filename: str,
    output_dir: Path,
    fit_result: FitResult,
):
    formalism_alias = reaction.formalism[:4]
    if sympy_model.max_complexity is None:
        lambdify_type = "normal"
    else:
        lambdify_type = "optimized"
    filename = filename.format(formalism_alias, lambdify_type)
    expected = fit_result.parameter_values
    imported = callback_type.load_latest_parameters(output_dir / filename)
    for par in expected:
        assert pytest.approx(expected[par], rel=1e-2) == imported[par]
