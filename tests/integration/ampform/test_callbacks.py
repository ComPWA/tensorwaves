from pathlib import Path
from typing import Type

import pytest
import qrules

from tensorwaves.interface import FitResult
from tensorwaves.optimizer.callbacks import CSVSummary, Loadable, YAMLSummary


@pytest.mark.parametrize(
    ("callback_type", "filename"),
    [
        (CSVSummary, "fit_traceback_{}.csv"),
        (YAMLSummary, "fit_result_{}.yml"),
    ],
)
def test_load_latest_parameters(
    reaction: qrules.ReactionInfo,
    callback_type: Type[Loadable],
    filename: str,
    output_dir: Path,
    fit_result: FitResult,
):
    formalism = reaction.formalism[:4]
    filename = filename.format(formalism)
    expected = fit_result.parameter_values
    imported = callback_type.load_latest_parameters(output_dir / filename)
    for par in expected:
        assert pytest.approx(expected[par], rel=1e-2) == imported[par]
