"""Integration tests: identification must recover the model's own parameters."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_sysid.export import export_updated_xml
from robot_sysid.parser import load_robot_model
from robot_sysid.simulator import collect_sysid_data
from robot_sysid.solver import identify_parameters
from robot_sysid.trajectory import generate_excitation_trajectory

XML_PATH = str(Path(__file__).parent / "models" / "arm7.xml")

# Ground truth for link7 from the XML
TRUE_MASS = 0.58
TRUE_COM = np.array([-9.2e-05, -0.079, 0.0029])


@pytest.mark.mujoco
@pytest.mark.parametrize("include_friction", [True, False])
def test_recovers_terminal_link_parameters(include_friction, tmp_path):
    model = load_robot_model(XML_PATH)
    trajectory = generate_excitation_trajectory(
        model=model, duration=5.0, sample_rate=500, seed=42
    )
    data = collect_sysid_data(
        xml_path=XML_PATH,
        trajectory=trajectory,
        terminal_joint=model.terminal_joint,
        terminal_body_name=model.terminal_body.name,
        include_friction=include_friction,
    )
    params = identify_parameters(
        data=data,
        joint_name=model.terminal_joint.name,
        include_friction=include_friction,
    )

    assert params.n_base_params == data.Y.shape[1]
    assert abs(params.mass - TRUE_MASS) < 0.05
    np.testing.assert_allclose(params.com_moment, TRUE_MASS * TRUE_COM, atol=1e-3)
    assert params.rmse < 0.1
    assert (params.coulomb_friction is not None) == include_friction

    # Exported model loads and reproduces the original inertial
    out_path = tmp_path / "updated_model.xml"
    export_updated_xml(XML_PATH, params, str(out_path))
    original = mujoco.MjModel.from_xml_path(XML_PATH)
    updated = mujoco.MjModel.from_xml_path(str(out_path))
    body_id = mujoco.mj_name2id(updated, mujoco.mjtObj.mjOBJ_BODY, "link7")
    np.testing.assert_allclose(updated.body_mass[body_id], TRUE_MASS, atol=1e-3)
    np.testing.assert_allclose(updated.body_ipos[body_id], TRUE_COM, atol=1e-4)
    np.testing.assert_allclose(
        updated.body_inertia[body_id], original.body_inertia[body_id], atol=1e-5
    )
