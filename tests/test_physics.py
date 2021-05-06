# pylint: disable=redefined-outer-name
import pytest
from ampform.helicity import HelicityModel

from tensorwaves.interfaces import DataSample
from tensorwaves.model import LambdifiedFunction
from tensorwaves.physics import create_intensity_component


def test_create_intensity_component(
    phsp_set: DataSample,
    es_helicity_model: HelicityModel,
    intensity: LambdifiedFunction,
):
    # pylint: disable=line-too-long
    model = es_helicity_model
    from_amplitudes = create_intensity_component(
        model,
        components=[
            R"A[J/\psi(1S)_{+1} \to f_{0}(500)_{0} \gamma_{+1}; f_{0}(500)_{0} \to \pi^{0}_{0} \pi^{0}_{0}]",
            R"A[J/\psi(1S)_{+1} \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980)_{0} \to \pi^{0}_{0} \pi^{0}_{0}]",
        ],
        backend="numpy",
    )
    from_intensity = create_intensity_component(
        model,
        components=R"I[J/\psi(1S)_{+1} \to \gamma_{+1} \pi^{0}_{0} \pi^{0}_{0}]",
        backend="numpy",
    )
    assert pytest.approx(from_amplitudes(phsp_set)) == from_intensity(phsp_set)

    intensity_components = [
        create_intensity_component(model, component, backend="numpy")
        for component in model.components
        if component.startswith("I")
    ]
    sub_intensities = [i(phsp_set) for i in intensity_components]
    assert pytest.approx(sum(sub_intensities)) == intensity(phsp_set)
