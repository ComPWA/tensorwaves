from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _pytest.config import Config as PytestConfig

if TYPE_CHECKING:
    from qrules import ParticleCollection


@pytest.fixture(scope="session")
def pdg() -> "ParticleCollection":
    # pylint: disable=import-outside-toplevel
    from qrules.particle import load_pdg

    return load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig: PytestConfig) -> Path:
    path = Path(f"{pytestconfig.rootpath}/tests/output")
    path.mkdir(exist_ok=True)
    return path
