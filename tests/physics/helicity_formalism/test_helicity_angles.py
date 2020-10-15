import expertsystem.amplitude.model as es
import numpy as np
import pytest

from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    SubSystem,
)

TEST_DATA = {
    "events": {
        0: [
            (0.514208, -0.184219, 1.23296, 1.35527),
            (0.0727385, -0.0528868, 0.826163, 0.841933),
            (-0.162529, 0.29976, -0.411133, 0.550927),
            (0.0486171, 0.151922, 0.370309, 0.425195),
            (-0.0555915, -0.100214, -0.0597338, 0.186869),
            (0.238921, 0.266712, -1.20442, 1.26375),
            (0.450724, -0.439515, -0.360076, 0.737698),
            (0.552298, 0.440006, 0.644927, 0.965809),
            (-0.248155, -0.158587, -0.229673, 0.397113),
            (1.33491, 0.358535, 0.0457548, 1.38955),
        ],
        1: [
            (-0.305812, 0.284, -0.630057, 0.755744),
            (0.784483, 0.614347, -0.255334, 1.02861),
            (-0.20767, 0.272796, 0.0990739, 0.356875),
            (0.404557, 0.510467, -0.276426, 0.70757),
            (0.47713, 0.284575, -0.775431, 0.953902),
            (-0.204775, -0.0197981, 0.0799868, 0.220732),
            (0.00590727, 0.709346, -0.190877, 0.734602),
            (0.329157, -0.431973, 0.272873, 0.607787),
            (-0.201436, -0.534829, 0.256253, 0.626325),
            (-0.196357, 0.00211926, -0.33282, 0.386432),
        ],
        2: [
            (-0.061663, -0.0211864, 0.144596, 0.208274),
            (-0.243319, -0.283044, -0.234866, 0.461193),
            (0.82872, -0.0465425, -0.599834, 1.03294),
            (0.263003, -0.089236, 0.686187, 0.752466),
            (0.656892, -0.107848, 0.309898, 0.746588),
            (0.521569, -0.0448683, 0.43283, 0.692537),
            (-0.517582, -0.676002, -0.0734335, 0.865147),
            (-0.975278, -0.0207817, -0.934467, 1.35759),
            (-0.41665, 0.237646, 0.691269, 0.852141),
            (-0.464203, -0.358114, 0.13307, 0.616162),
        ],
        3: [
            (-0.146733, -0.0785946, -0.747499, 0.777613),
            (-0.613903, -0.278416, -0.335962, 0.765168),
            (-0.458522, -0.526014, 0.911894, 1.15616),
            (-0.716177, -0.573154, -0.780069, 1.21167),
            (-1.07843, -0.0765127, 0.525267, 1.20954),
            (-0.555715, -0.202046, 0.691605, 0.919879),
            (0.0609506, 0.406171, 0.624387, 0.759452),
            (0.0938229, 0.012748, 0.0166676, 0.165716),
            (0.866241, 0.455769, -0.717849, 1.22132),
            (-0.674348, -0.0025409, 0.153994, 0.704759),
        ],
    },
    "angles": {
        (((1, 2, 3), (0,)), (), ()): [
            (-0.914298, 2.79758),
            (-0.994127, 2.51292),
            (0.769715, -1.07396),
            (-0.918418, -1.88051),
            (0.462214, 1.06433),
            (0.958535, -2.30129),
            (0.496489, 2.36878),
            (-0.674376, -2.46888),
            (0.614968, 0.568649),
            (-0.0330843, -2.8792),
        ],
        (((2, 3), (1,)), (0,), ()): [
            (-0.772533, 1.04362),
            (0.163659, 1.87349),
            (0.556365, 0.160733),
            (0.133251, -2.81088),
            (-0.0264361, 2.84379),
            (0.227188, 2.29128),
            (-0.166924, 2.24539),
            (0.652761, -1.20272),
            (0.443122, 0.615838),
            (0.503577, 2.98067),
        ],
        (((2,), (3,)), (1,), (0,)): [
            (0.460324, -2.77203),
            (-0.410464, 1.45339),
            (0.248566, -2.51096),
            (-0.301959, 2.71085),
            (-0.522502, -1.12706),
            (0.787267, -3.01323),
            (0.488066, 2.07305),
            (0.954167, 0.502648),
            (-0.553114, -1.23689),
            (0.00256349, 1.7605),
        ],
    },
}


@pytest.mark.parametrize(
    "test_events, expected_angles",
    [(TEST_DATA["events"], TEST_DATA["angles"])],
)  # pylint: disable=too-many-locals
def test_helicity_angles_correctness(test_events, expected_angles, pdg):
    kinematics = es.Kinematics(particles=pdg)
    kinematics.set_reaction(
        initial_state=["J/psi(1S)"],
        final_state=["pi0", "gamma", "pi0", "pi0"],
        intermediate_states=-1,
    )
    model = es.AmplitudeModel(
        particles=pdg,
        kinematics=kinematics,
        parameters=None,  # type: ignore
        intensity=None,  # type: ignore
        dynamics=None,  # type: ignore
    )
    kin = HelicityKinematics.from_model(model)
    subsys_angle_names = {}
    for subsys in expected_angles.keys():
        temp_names = kin.register_subsystem(SubSystem(*subsys))
        subsys_angle_names.update({subsys: [temp_names[1], temp_names[2]]})

    data = np.array(tuple(np.array(v) for v in test_events.values()))
    kinematic_vars = kin.convert(data)

    assert len(kinematic_vars) == 3 * len(expected_angles.keys())
    number_of_events = len(data[0])
    for subsys, angle_names in subsys_angle_names.items():
        for name in angle_names:
            assert len(kinematic_vars[name]) == number_of_events

        expected_values = np.array(np.array(expected_angles[subsys]).T)
        # test cos(theta)
        np.testing.assert_array_almost_equal(
            np.cos(kinematic_vars[angle_names[0]]), expected_values[0], 1e-6
        )
        # test phi
        if subsys == (((2,), (3,)), (1,), (0,)):
            for kin_var, expected in zip(
                kinematic_vars[angle_names[1]], expected_values[1]
            ):
                assert round(kin_var, 4) == round(
                    expected - np.pi, 4
                ) or round(kin_var, 4) == round(expected + np.pi, 4)
        else:
            np.testing.assert_array_almost_equal(
                kinematic_vars[angle_names[1]], expected_values[1], 1e-6
            )
