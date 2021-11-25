import pytest
import qrules
from _pytest.config import Config as PytestConfig


@pytest.fixture(scope="session")
def pdg() -> qrules.ParticleCollection:
    return qrules.particle.load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig: PytestConfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"
