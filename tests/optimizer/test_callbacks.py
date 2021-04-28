from typing import Type

import pytest

from tensorwaves.interfaces import FitResult
from tensorwaves.optimizer.callbacks import CSVSummary, Loadable, YAMLSummary


@pytest.mark.parametrize(
    ("callback_type", "filename"),
    [
        (CSVSummary, "fit_traceback.csv"),
        (YAMLSummary, "fit_result.yml"),
    ],
)
def test_load_latest_parameters(
    callback_type: Type[Loadable],
    filename: str,
    output_dir: str,
    fit_result: FitResult,
):
    expected = fit_result.parameter_values
    imported = callback_type.load_latest_parameters(output_dir + filename)
    for par in expected:
        assert pytest.approx(expected[par], rel=1e-2) == imported[par]
