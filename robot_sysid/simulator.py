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
from robot_sysid.regressor import (
    compute_joint_regressor_row,
    compute_joint_regressor_row_with_friction,
)
from robot_sysid.trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class RegressorData:
    """Collected regressor matrix and torque data from simulation.

    Attributes:
        Y: Regressor matrix of shape (n_samples, 10) or (n_samples, 12).
        tau: Torque vector of shape (n_samples,).
        time: Timestamps of shape (n_samples,).
        condition_number: Condition number of the regressor matrix.
        matrix_rank: Rank of the regressor matrix.
    """

    Y: np.ndarray  # (n_samples, 10) or (n_samples, 12)
    tau: np.ndarray  # (n_samples,)
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
    2. Computes Jacobians at the terminal joint site
    3. Computes body-frame velocity and acceleration using finite-difference
       Jdot and gravity addition
    4. Builds a regressor row from the kinematic data
    5. Records the inverse dynamics torque

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

    # Find terminal body and site IDs
    terminal_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, terminal_body_name
    )
    if terminal_body_id < 0:
        raise ValueError(
            f"Terminal body '{terminal_body_name}' not found in model"
        )

    terminal_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, terminal_joint.site_name
    )
    if terminal_site_id < 0:
        raise ValueError(
            f"Terminal site '{terminal_joint.site_name}' not found in model"
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

    # Joint axis in body frame
    joint_axis = terminal_joint.axis

    # Gravity vector in world frame
    g_world = np.array([0.0, 0.0, 9.81])

    # Allocate Jacobian arrays
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Finite difference epsilon for Jdot computation
    eps = 1e-6

    n_samples = len(trajectory.time)
    n_cols = 12 if include_friction else 10
    Y = np.zeros((n_samples, n_cols))
    tau = np.zeros(n_samples)

    # Collect data along the trajectory
    for i in range(n_samples):
        qpos = trajectory.position[i, :]
        qvel = trajectory.velocity[i, :]
        qacc = trajectory.acceleration[i, :]

        # Set state
        data.qpos[: trajectory.n_joints] = qpos
        data.qvel[: trajectory.n_joints] = qvel
        data.qacc[: trajectory.n_joints] = qacc

        # Compute kinematics
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)

        # Get Jacobian at terminal site
        mujoco.mj_jacSite(model, data, jacp, jacr, terminal_site_id)

        # Compute velocity in world frame
        v_world = jacp @ qvel
        w_world = jacr @ qvel

        # Rotation matrix from world to body frame
        R = data.xmat[terminal_body_id].reshape(3, 3)

        # Transform velocity to body frame
        w_local = R.T @ w_world
        v_local = R.T @ v_world

        # Compute Jdot @ qvel via finite difference
        # Perturb qpos by eps * qvel and recompute Jacobian
        data.qpos[: trajectory.n_joints] = qpos + eps * qvel
        mujoco.mj_kinematics(model, data)

        jacp_plus = np.zeros((3, model.nv))
        jacr_plus = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp_plus, jacr_plus, terminal_site_id)

        Jdot_v_linear = (jacp_plus - jacp) @ qvel / eps
        Jdot_v_angular = (jacr_plus - jacr) @ qvel / eps

        # Full acceleration in world frame
        dv_world = jacp @ qacc + Jdot_v_linear
        dw_world = jacr @ qacc + Jdot_v_angular

        # Add gravity to linear acceleration before transforming to body frame
        dv_world_with_g = dv_world + g_world

        # Transform acceleration to body frame
        dw_local = R.T @ dw_world
        dv_local = R.T @ dv_world_with_g - np.cross(w_local, v_local)

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

        # Compute regressor row
        if include_friction:
            # Get joint velocity for friction terms
            joint_velocity = qvel[terminal_dof_idx]
            Y_row = compute_joint_regressor_row_with_friction(
                w_local[0],
                w_local[1],
                w_local[2],
                v_local[0],
                v_local[1],
                v_local[2],
                dw_local[0],
                dw_local[1],
                dw_local[2],
                dv_local[0],
                dv_local[1],
                dv_local[2],
                joint_axis,
                joint_velocity,
            )
        else:
            Y_row = compute_joint_regressor_row(
                w_local[0],
                w_local[1],
                w_local[2],
                v_local[0],
                v_local[1],
                v_local[2],
                dw_local[0],
                dw_local[1],
                dw_local[2],
                dv_local[0],
                dv_local[1],
                dv_local[2],
                joint_axis,
            )

        # Restore state and compute inverse dynamics
        data.qpos[: trajectory.n_joints] = qpos
        data.qvel[: trajectory.n_joints] = qvel
        data.qacc[: trajectory.n_joints] = qacc
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)
        mujoco.mj_inverse(model, data)

        # Extract torque for terminal joint
        tau_mj = data.qfrc_inverse[terminal_dof_idx]

        # Check for NaN in torque
        if np.isnan(tau_mj):
            raise RuntimeError(
                f"Simulation diverged at timestep {i} (t={trajectory.time[i]:.3f}s): "
                f"NaN detected in inverse dynamics torque"
            )

        Y[i, :] = Y_row
        tau[i] = tau_mj

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
