"""Model parser for MuJoCo XML robot descriptions.

Loads a MuJoCo XML file, validates it, and extracts the robot's kinematic
structure including joints, bodies, and inertial parameters.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import defusedxml.ElementTree as SafeET

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JointInfo:
    """Metadata for a single revolute joint."""

    name: str
    axis: np.ndarray  # 3D unit vector in parent body frame
    body_name: str  # body this joint belongs to
    site_name: str  # site at joint origin (for Jacobian computation)
    range: tuple[float, float]  # (lower, upper) position limits in radians
    force_range: tuple[float, float]  # (min, max) actuator force limits


@dataclass
class BodyInfo:
    """Inertial parameters for a single body."""

    name: str
    mass: float
    com: np.ndarray  # center of mass [x, y, z]
    inertia: np.ndarray  # [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] (full inertia)
    parent: str | None  # parent body name


@dataclass
class RobotModel:
    """Complete robot kinematic and inertial model."""

    xml_path: str
    joints: list[JointInfo]
    bodies: list[BodyInfo]
    terminal_joint: JointInfo
    terminal_body: BodyInfo
    n_joints: int


def _get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Get the name of a body by its ID."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _get_joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Get the name of a joint by its ID."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)


def _find_site_for_joint(
    model: mujoco.MjModel, joint_name: str, body_name: str
) -> str:
    """Find a site at the joint origin, or fall back to body name convention.

    MuJoCo models may have sites named after joints (e.g., "joint1" site for
    "joint1" joint). If no matching site exists, we use the body name as a
    convention for downstream code to handle.
    """
    # Try to find a site with the same name as the joint
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, joint_name)
    if site_id >= 0:
        return joint_name

    # Fall back to body name as convention
    return body_name


def _get_actuator_force_range(
    model: mujoco.MjModel, joint_name: str
) -> tuple[float, float]:
    """Get the actuator force range for a joint.

    Searches through actuators to find one that drives the given joint.
    Returns (0.0, 0.0) if no actuator is found or if force limits are not set.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return (0.0, 0.0)

    for act_id in range(model.nu):
        # Check if this actuator drives our joint via transmission
        # MuJoCo actuators have trnid which points to the joint
        if model.actuator_trntype[act_id] == 0:  # mjTRN_JOINT
            if model.actuator_trnid[act_id, 0] == joint_id:
                # Check forcerange first, then ctrlrange
                frc_range = model.actuator_forcerange[act_id]
                if frc_range[0] != 0.0 or frc_range[1] != 0.0:
                    return (float(frc_range[0]), float(frc_range[1]))

                ctrl_range = model.actuator_ctrlrange[act_id]
                if ctrl_range[0] != 0.0 or ctrl_range[1] != 0.0:
                    return (float(ctrl_range[0]), float(ctrl_range[1]))

                return (0.0, 0.0)

    return (0.0, 0.0)


def _extract_full_inertia(
    model: mujoco.MjModel, body_id: int
) -> np.ndarray:
    """Extract the full 6-element inertia vector [Ixx, Ixy, Ixz, Iyy, Iyz, Izz].

    MuJoCo stores diagonal inertia in model.body_inertia as [Ixx, Iyy, Izz].
    The off-diagonal terms are zero in the body's principal inertia frame.

    To get the full inertia tensor in the body frame, we need to rotate the
    diagonal inertia from the principal frame using body_iquat (the quaternion
    from body frame to inertia frame).

    Returns [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] in the body frame.
    """
    # Diagonal inertia in principal frame
    diag = model.body_inertia[body_id]  # [Ixx, Iyy, Izz] in principal frame

    # Inertia frame quaternion (body-to-inertia rotation)
    iquat = model.body_iquat[body_id]

    # Convert quaternion to rotation matrix
    # MuJoCo quaternion format: [w, x, y, z]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, iquat)
    R = R.reshape(3, 3)

    # Diagonal inertia tensor in principal frame
    I_principal = np.diag(diag)

    # Rotate to body frame: I_body = R @ I_principal @ R^T
    I_body = R @ I_principal @ R.T

    # Extract [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    return np.array([
        I_body[0, 0],  # Ixx
        I_body[0, 1],  # Ixy
        I_body[0, 2],  # Ixz
        I_body[1, 1],  # Iyy
        I_body[1, 2],  # Iyz
        I_body[2, 2],  # Izz
    ])


def _walk_body_tree(model: mujoco.MjModel) -> tuple[list[JointInfo], list[BodyInfo]]:
    """Walk the MuJoCo body tree and extract joints and bodies.

    Traverses all bodies (skipping worldbody at index 0) and extracts
    joint information and body inertial parameters.

    Returns:
        Tuple of (joints, bodies) lists ordered by depth in the kinematic chain.
    """
    joints: list[JointInfo] = []
    bodies: list[BodyInfo] = []

    # Iterate over all bodies (skip worldbody at index 0)
    for body_id in range(1, model.nbody):
        body_name = _get_body_name(model, body_id)
        if body_name is None:
            continue

        # Get parent body name
        parent_id = model.body_parentid[body_id]
        parent_name: str | None = None
        if parent_id > 0:  # 0 is worldbody
            parent_name = _get_body_name(model, parent_id)
        elif parent_id == 0:
            parent_name = "world"

        # Extract body inertial parameters
        mass = float(model.body_mass[body_id])
        com = model.body_ipos[body_id].copy()  # CoM in body frame
        inertia = _extract_full_inertia(model, body_id)

        bodies.append(BodyInfo(
            name=body_name,
            mass=mass,
            com=com,
            inertia=inertia,
            parent=parent_name,
        ))

        # Extract joints belonging to this body
        # MuJoCo: body_jntadr[body_id] is the first joint index,
        # body_jntnum[body_id] is the number of joints
        jnt_start = model.body_jntadr[body_id]
        jnt_count = model.body_jntnum[body_id]

        for j in range(jnt_count):
            jnt_id = jnt_start + j

            # Only handle revolute (hinge) joints
            if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue

            jnt_name = _get_joint_name(model, jnt_id)
            if jnt_name is None:
                continue

            axis = model.jnt_axis[jnt_id].copy()
            jnt_range = (float(model.jnt_range[jnt_id, 0]),
                         float(model.jnt_range[jnt_id, 1]))
            force_range = _get_actuator_force_range(model, jnt_name)
            site_name = _find_site_for_joint(model, jnt_name, body_name)

            joints.append(JointInfo(
                name=jnt_name,
                axis=axis,
                body_name=body_name,
                site_name=site_name,
                range=jnt_range,
                force_range=force_range,
            ))

    return joints, bodies


def _find_terminal_joint(
    joints: list[JointInfo], bodies: list[BodyInfo]
) -> tuple[JointInfo, BodyInfo]:
    """Find the terminal (deepest) joint in the kinematic chain.

    The terminal joint is the one whose body is deepest in the parent-child
    hierarchy. We trace each joint's body back to the root and pick the one
    with the longest chain.

    Returns:
        Tuple of (terminal_joint, terminal_body).

    Raises:
        ValueError: If no joints are found.
    """
    if not joints:
        raise ValueError("No revolute joints found in model")

    # Build a name -> BodyInfo lookup
    body_map = {b.name: b for b in bodies}

    def chain_depth(body_name: str) -> int:
        """Count the depth of a body in the kinematic chain."""
        depth = 0
        current = body_name
        while current in body_map and body_map[current].parent is not None:
            depth += 1
            parent = body_map[current].parent
            if parent == "world" or parent not in body_map:
                break
            current = parent
        return depth

    # Find the joint with the deepest body
    max_depth = -1
    terminal_joint = joints[0]
    terminal_body = body_map[joints[0].body_name]

    for joint in joints:
        depth = chain_depth(joint.body_name)
        if depth > max_depth:
            max_depth = depth
            terminal_joint = joint
            terminal_body = body_map[joint.body_name]

    return terminal_joint, terminal_body


def _validate_meshes(xml_path: str, mesh_dir: str) -> None:
    """Validate that all mesh files referenced in the XML exist.

    Parses the XML to find mesh file references and checks that each
    file exists in the specified mesh directory.

    Args:
        xml_path: Path to the MuJoCo XML file.
        mesh_dir: Path to the directory containing mesh files.

    Raises:
        FileNotFoundError: If any referenced mesh files are missing.
    """
    try:
        tree = SafeET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file '{xml_path}': {e}") from e

    root = tree.getroot()

    # Find all mesh elements in the asset section
    missing_files: list[str] = []
    for mesh_elem in root.iter("mesh"):
        mesh_file = mesh_elem.get("file")
        if mesh_file is not None:
            full_path = os.path.join(mesh_dir, mesh_file)
            if not os.path.isfile(full_path):
                missing_files.append(full_path)

    if missing_files:
        missing_list = "\n  ".join(missing_files)
        raise FileNotFoundError(
            f"Missing mesh files:\n  {missing_list}"
        )


def load_robot_model(
    xml_path: str, mesh_dir: str | None = None
) -> RobotModel:
    """Load and validate a MuJoCo XML, return structured RobotModel.

    Args:
        xml_path: Path to the MuJoCo XML file.
        mesh_dir: Optional path to the mesh directory. If provided, validates
            that all referenced mesh files exist.

    Returns:
        A RobotModel containing the robot's kinematic structure and
        inertial parameters.

    Raises:
        ValueError: If the XML is malformed or contains no revolute joints.
        FileNotFoundError: If the XML file doesn't exist or mesh files are missing.
    """
    # Validate file exists
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML file not found: '{xml_path}'")

    # Validate mesh files if mesh_dir is provided
    if mesh_dir is not None:
        _validate_meshes(xml_path, mesh_dir)

    # Load the MuJoCo model — this validates the XML structure
    try:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        raise ValueError(
            f"Failed to load MuJoCo model from '{xml_path}': {e}"
        ) from e

    # Walk the body tree to extract joints and bodies
    joints, bodies = _walk_body_tree(mj_model)

    if not joints:
        raise ValueError("No revolute joints found in model")

    # Find the terminal (deepest) joint
    terminal_joint, terminal_body = _find_terminal_joint(joints, bodies)

    return RobotModel(
        xml_path=xml_path,
        joints=joints,
        bodies=bodies,
        terminal_joint=terminal_joint,
        terminal_body=terminal_body,
        n_joints=len(joints),
    )
