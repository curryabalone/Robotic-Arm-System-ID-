"""MuJoCo simulation and data collection for system identification.

Runs a MuJoCo simulation along an excitation trajectory and collects regressor
matrix and torque data for parameter identification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from robot_sysid.parser import JointInfo
from robot_sysid.regressor import get_regressor_func
from robot_sysid.trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class RegressorData:
    """Collected regressor matrix and torque data from simulation.

    Attributes:
        Y: Regressor matrix of shape (n_rows, 10) or (n_rows, 12), with one
            row per supporting joint per sample.
        tau: Torque vector of shape (n_rows,).
        time: Timestamps of shape (n_samples,).
        condition_number: Condition number of the regressor matrix.
        matrix_rank: Rank of the regressor matrix.
    """

    Y: np.ndarray  # (n_rows, 10) or (n_rows, 12)
    tau: np.ndarray  # (n_rows,)
    time: np.ndarray  # (n_samples,)
    condition_number: float
    matrix_rank: int


def collect_sysid_data(
    xml_path: str,
    trajectory: Trajectory,
    terminal_joint: JointInfo,
    terminal_body_name: str,
    include_friction: bool = True,
) -> RegressorData:
    """Run MuJoCo simulation and collect regressor + torque data.

    This function loads the MuJoCo model, runs a simulation along the provided
    trajectory, and collects the regressor matrix Y and torque vector τ such
    that τ = Y × θ, where θ is the vector of inertial parameters (and
    optionally friction coefficients).

    The simulation uses MuJoCo's inverse dynamics (`mj_inverse`) to compute
    ground-truth torques. At each timestep, the function:
    1. Sets qpos, qvel, qacc from the trajectory
    2. Runs inverse dynamics
    3. Reads the terminal body's velocity, acceleration (with gravity) and
       Jacobian at its frame origin
    4. Builds one regressor row per joint supporting the terminal body
    5. Records the inverse dynamics torques

    The terminal joint torque alone cannot observe the mass (it only sees the
    moment about its axis), so the upstream joints are used as well. Their
    torques are reduced to the terminal body's contribution by subtracting
    inverse dynamics of the same model with that body's inertia removed.

    Args:
        xml_path: Path to the MuJoCo XML model file.
        trajectory: Excitation trajectory with position, velocity, and
            acceleration arrays.
        terminal_joint: Information about the terminal joint (name, axis, site).
        terminal_body_name: Name of the terminal body in the MuJoCo model.
        include_friction: Whether to include friction terms in the regressor
            (default True). If True, regressor is 12-column; if False, 10-column.

    Returns:
        RegressorData containing the stacked regressor matrix Y, torque vector
        τ, timestamps, condition number, and matrix rank.

    Raises:
        RuntimeError: If the simulation produces NaN values or diverges.
        ValueError: If the terminal joint or body cannot be found in the model.
    """
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Find terminal body ID
    terminal_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, terminal_body_name
    )
    if terminal_body_id < 0:
        raise ValueError(
            f"Terminal body '{terminal_body_name}' not found in model"
        )

    # Find terminal joint ID for torque extraction
    terminal_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, terminal_joint.name
    )
    if terminal_joint_id < 0:
        raise ValueError(
            f"Terminal joint '{terminal_joint.name}' not found in model"
        )

    # Get the DOF address for this joint (for indexing qvel and qfrc_inverse)
    terminal_dof_idx = model.jnt_dofadr[terminal_joint_id]

    # DOFs of all joints between the world and the terminal body
    chain_dofs = []
    body_id = terminal_body_id
    while body_id > 0:
        jnt_start = model.body_jntadr[body_id]
        for jnt_id in range(jnt_start, jnt_start + model.body_jntnum[body_id]):
            chain_dofs.append(model.jnt_dofadr[jnt_id])
        body_id = model.body_parentid[body_id]
    chain_dofs = sorted(chain_dofs)
    n_rows = len(chain_dofs)
    terminal_row = chain_dofs.index(terminal_dof_idx)

    # Same model without the terminal body's inertia, to isolate its torques
    model_wo = mujoco.MjModel.from_xml_path(xml_path)
    model_wo.body_mass[terminal_body_id] = 0.0
    model_wo.body_inertia[terminal_body_id] = 0.0
    data_wo = mujoco.MjData(model_wo)

    regressor_func = get_regressor_func()

    # Allocate Jacobian and body motion arrays
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    vel = np.zeros(6)
    acc = np.zeros(6)

    n_samples = len(trajectory.time)
    n_cols = 12 if include_friction else 10
    Y = np.zeros((n_samples * n_rows, n_cols))
    tau = np.zeros(n_samples * n_rows)

    # Collect data along the trajectory
    for i in range(n_samples):
        qpos = trajectory.position[i, :]
        qvel = trajectory.velocity[i, :]
        qacc = trajectory.acceleration[i, :]

        # Inverse dynamics with and without the terminal body
        for m, d in ((model, data), (model_wo, data_wo)):
            d.qpos[: trajectory.n_joints] = qpos
            d.qvel[: trajectory.n_joints] = qvel
            d.qacc[: trajectory.n_joints] = qacc
            mujoco.mj_inverse(m, d)
        mujoco.mj_rnePostConstraint(model, data)

        # Exact body-frame velocity and acceleration (gravity included as an
        # upward acceleration) of the terminal body origin
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, terminal_body_id, vel, 1
        )
        mujoco.mj_objectAcceleration(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, terminal_body_id, acc, 1
        )
        w_local, v_local = vel[:3], vel[3:]
        dw_local = acc[:3]
        # Classical -> spatial acceleration, as used by the regressor
        dv_local = acc[3:] - np.cross(w_local, v_local)

        # Jacobian at the terminal body origin, rotated into the body frame
        mujoco.mj_jacBody(model, data, jacp, jacr, terminal_body_id)
        R = data.xmat[terminal_body_id].reshape(3, 3)

        # Check for NaN or divergence
        if np.any(np.isnan(w_local)) or np.any(np.isnan(v_local)):
            raise RuntimeError(
                f"Simulation diverged at timestep {i} (t={trajectory.time[i]:.3f}s): "
                f"NaN detected in velocity"
            )
        if np.any(np.isnan(dw_local)) or np.any(np.isnan(dv_local)):
            raise RuntimeError(
                f"Simulation diverged at timestep {i} (t={trajectory.time[i]:.3f}s): "
                f"NaN detected in acceleration"
            )

        # Body wrench regressor (6x10), projected onto each chain joint's
        # motion subspace expressed in the body frame
        Y_body = np.asarray(
            regressor_func(*w_local, *v_local, *dw_local, *dv_local), dtype=float
        )
        S = np.vstack([R.T @ jacr[:, chain_dofs], R.T @ jacp[:, chain_dofs]])
        rows = slice(i * n_rows, (i + 1) * n_rows)
        Y[rows, :10] = S.T @ Y_body

        # Friction only acts on the terminal joint
        if include_friction:
            joint_velocity = qvel[terminal_dof_idx]
            Y[i * n_rows + terminal_row, 10:] = [np.sign(joint_velocity), joint_velocity]

        # Upstream torques: terminal body's contribution only.
        # Terminal joint: full torque, which also carries its friction.
        tau_mj = data.qfrc_inverse[chain_dofs] - data_wo.qfrc_inverse[chain_dofs]
        tau_mj[terminal_row] = data.qfrc_inverse[terminal_dof_idx]

        # Check for NaN in torque
        if np.any(np.isnan(tau_mj)):
            raise RuntimeError(
                f"Simulation diverged at timestep {i} (t={trajectory.time[i]:.3f}s): "
                f"NaN detected in inverse dynamics torque"
            )

        tau[rows] = tau_mj

        # Progress reporting every 2000 samples
        if i % 2000 == 0 and i > 0:
            logger.info(
                "Collecting data: %d/%d (%.0f%%)", i, n_samples, 100 * i / n_samples
            )

    # Compute condition number and rank
    condition_number = float(np.linalg.cond(Y))
    matrix_rank = int(np.linalg.matrix_rank(Y))

    return RegressorData(
        Y=Y,
        tau=tau,
        time=trajectory.time.copy(),
        condition_number=condition_number,
        matrix_rank=matrix_rank,
    )
