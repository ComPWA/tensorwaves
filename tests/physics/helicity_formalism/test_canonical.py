import math

import expertsystem.amplitude.model as es
import pytest  # type: ignore

from tensorwaves.physics.helicity_formalism.amplitude import (
    _clebsch_gordan_coefficient,
    _determine_canonical_prefactor,
)


@pytest.mark.parametrize(
    "clebsch_gordan, expected_value",
    [
        (
            es.ClebschGordan(
                J=1.0,
                M=1.0,
                j_1=0.5,
                m_1=0.5,
                j_2=0.5,
                m_2=0.5,
            ),
            1,
        ),
        (
            es.ClebschGordan(
                J=1.0,
                M=0.0,
                j_1=0.5,
                m_1=0.5,
                j_2=0.5,
                m_2=-0.5,
            ),
            math.sqrt(1 / 2),
        ),
        (
            es.ClebschGordan(
                J=1.0,
                M=0.0,
                j_1=0.5,
                m_1=-0.5,
                j_2=0.5,
                m_2=0.5,
            ),
            math.sqrt(1 / 2),
        ),
        (
            es.ClebschGordan(
                J=0.0,
                M=0.0,
                j_1=0.5,
                m_1=-0.5,
                j_2=0.5,
                m_2=0.5,
            ),
            -math.sqrt(1 / 2),
        ),
        (
            es.ClebschGordan(
                J=0.0,
                M=0.0,
                j_1=0.5,
                m_1=0.5,
                j_2=0.5,
                m_2=-0.5,
            ),
            math.sqrt(1 / 2),
        ),
        (
            es.ClebschGordan(
                J=3.0,
                M=3.0,
                j_1=2.0,
                m_1=2.0,
                j_2=1.0,
                m_2=1.0,
            ),
            1,
        ),
        (
            es.ClebschGordan(
                J=3.0,
                M=2.0,
                j_1=2.0,
                m_1=2.0,
                j_2=1.0,
                m_2=0.0,
            ),
            math.sqrt(1 / 3),
        ),
        (
            es.ClebschGordan(
                J=1.0,
                M=1.0,
                j_1=2.0,
                m_1=0.0,
                j_2=1.0,
                m_2=1.0,
            ),
            math.sqrt(1 / 10),
        ),
    ],
)
def test_clebsch_gordan_coefficient(
    clebsch_gordan: es.ClebschGordan, expected_value: float
):
    cgc = _clebsch_gordan_coefficient(clebsch_gordan)
    assert cgc == pytest.approx(expected_value, rel=1e-6)


@pytest.mark.parametrize(
    "cano_decay, expected_value",
    [
        (
            es.CanonicalDecay(
                decaying_particle=None,  # type: ignore
                decay_products=None,  # type: ignore
                l_s=es.ClebschGordan(
                    J=1.0, M=1.0, j_1=2.0, m_1=0.0, j_2=1.0, m_2=1.0
                ),
                s2s3=es.ClebschGordan(
                    J=1.0, M=1.0, j_1=0.0, m_1=0.0, j_2=1.0, m_2=1.0
                ),
            ),
            math.sqrt(1 / 10) * 1,
        ),
        (
            es.CanonicalDecay(
                decaying_particle=None,  # type: ignore
                decay_products=None,  # type: ignore
                l_s=es.ClebschGordan(
                    J=1.0, M=1.0, j_1=2.0, m_1=0.0, j_2=1.0, m_2=1.0
                ),
                s2s3=es.ClebschGordan(
                    J=1.0, M=1.0, j_1=1.0, m_1=0.0, j_2=1.0, m_2=1.0
                ),
            ),
            math.sqrt(1 / 10) * -math.sqrt(1 / 2),
        ),
    ],
)
def test_determine_canonical_prefactor(
    cano_decay: es.CanonicalDecay, expected_value: float
):
    prefactor = _determine_canonical_prefactor(cano_decay)
    assert prefactor == pytest.approx(expected_value, rel=1e-6)
