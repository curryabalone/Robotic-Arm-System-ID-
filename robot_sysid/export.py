"""Export functions for system identification results.

Provides functions to export identified parameters to JSON, update MuJoCo XML
files with identified inertial parameters, and export trajectories for Damiao
motor playback.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import defusedxml.ElementTree as SafeET
import numpy as np

from robot_sysid.solver import IdentifiedParams
from robot_sysid.trajectory import Trajectory

logger = logging.getLogger(__name__)


# Damiao motor limits extracted from Ragtime Arm/damiao.py
# Format: {motor_type: (pos_max, vel_max, torque_max)}
# pos_max in radians, vel_max in rad/s, torque_max in N·m
DAMIAO_MOTOR_LIMITS: dict[str, tuple[float, float, float]] = {
    "DM3507": (12.566, 50.0, 5.0),
    "DM4310": (12.5, 30.0, 10.0),
    "DM4310_48V": (12.5, 50.0, 10.0),
    "DM4340": (12.5, 10.0, 28.0),
    "DM4340_48V": (12.5, 20.0, 28.0),
    "DM6006": (12.5, 45.0, 12.0),
    "DM6248": (12.566, 20.0, 120.0),
    "DM8006": (12.5, 45.0, 20.0),
    "DM8009": (12.5, 45.0, 54.0),
    "DM10010L": (12.5, 25.0, 200.0),
    "DM10010": (12.5, 20.0, 200.0),
    "DMH3510": (12.5, 280.0, 1.0),
    "DMH6215": (12.5, 45.0, 10.0),
    "DMS3519": (12.5, 2000.0, 2.0),
    "DMG6220": (12.5, 45.0, 10.0),
}


def export_params_json(params: IdentifiedParams, output_path: str) -> None:
    """Write identified parameters to JSON file.

    Serializes the IdentifiedParams object to a JSON file with pretty printing
    (indent=2).

    Args:
        params: Identified parameters to export.
        output_path: Path to the output JSON file.

    Raises:
        IOError: If the file cannot be written.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(params.to_dict(), f, indent=2)


def export_updated_xml(
    original_xml_path: str,
    params: IdentifiedParams,
    output_path: str,
) -> None:
    """Write updated MuJoCo XML with identified inertial parameters.

    Parses the original XML, finds the terminal body element, updates its
    inertial subelement with the identified mass, center of mass, and inertia
    tensor, and writes the updated XML to a new file. Adds a comment noting
    which body was identified and the identification RMSE.

    The inertia tensor is converted from the 6-element vector format
    [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] to the MuJoCo fullinertia format
    [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].

    Args:
        original_xml_path: Path to the original MuJoCo XML file.
        params: Identified parameters containing the inertial values.
        output_path: Path to the output XML file.

    Raises:
        FileNotFoundError: If the original XML file does not exist.
        ValueError: If the terminal body or inertial element cannot be found.
        IOError: If the output file cannot be written.
    """
    # Parse the original XML
    tree = SafeET.parse(original_xml_path)
    root = tree.getroot()

    # Find the terminal body by searching for a body with a joint matching
    # params.joint_name. We need to find the body that contains this joint.
    terminal_body = _find_body_with_joint(root, params.joint_name)

    if terminal_body is None:
        raise ValueError(
            f"Could not find body containing joint '{params.joint_name}' "
            f"in XML file '{original_xml_path}'"
        )

    body_name = terminal_body.get("name", "unknown")

    # Find or create the inertial element
    inertial = terminal_body.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(terminal_body, "inertial")

    # Compute center of mass from first moment: com = h / m
    com = params.com_moment / params.mass

    # Convert inertia from [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] to MuJoCo fullinertia
    # format [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    fullinertia = np.array([
        params.inertia[0],  # Ixx
        params.inertia[3],  # Iyy
        params.inertia[5],  # Izz
        params.inertia[1],  # Ixy
        params.inertia[2],  # Ixz
        params.inertia[4],  # Iyz
    ])

    # Update inertial attributes
    inertial.set("mass", f"{params.mass:.6f}")
    inertial.set("pos", f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
    inertial.set(
        "fullinertia",
        f"{fullinertia[0]:.6f} {fullinertia[1]:.6f} {fullinertia[2]:.6f} "
        f"{fullinertia[3]:.6f} {fullinertia[4]:.6f} {fullinertia[5]:.6f}"
    )

    # Add a comment before the body element noting the identification
    comment_text = (
        f" System ID: body '{body_name}' identified from joint '{params.joint_name}', "
        f"RMSE={params.rmse:.6f} N·m "
    )
    comment = ET.Comment(comment_text)

    # Insert comment before the terminal body
    parent = _find_parent(root, terminal_body)
    if parent is not None:
        parent_children = list(parent)
        idx = parent_children.index(terminal_body)
        parent.insert(idx, comment)

    # Write the updated XML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use ET.indent for pretty printing (Python 3.9+)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        # Fallback for Python < 3.9: write without indentation
        pass

    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def export_damiao_trajectory(
    trajectory: Trajectory,
    motor_types: list[str],
    can_ids: list[int],
    output_path: str,
) -> dict[str, float]:
    """Export trajectory CSV for Damiao motors.

    Writes a CSV file with timestamped rows containing position, velocity, and
    feedforward torque for each joint. Values are clamped to the motor limits
    for the specified motor types. The CSV includes header comments with motor
    type, CAN ID, and control mode information.

    The CSV format is:
    ```
    # motor_type=DM4340,can_id=1,mode=MIT
    # motor_type=DM4340,can_id=2,mode=MIT
    # ...
    time,j0_pos,j0_vel,j0_tau,j1_pos,j1_vel,j1_tau,...
    0.000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,...
    0.001,0.0012,0.3100,0.0500,0.0008,0.2200,0.0300,...
    ...
    ```

    Note: This function assumes feedforward torque is zero for all joints, as
    the trajectory generator does not compute torques. For actual hardware
    playback, torques should be computed from inverse dynamics.

    Args:
        trajectory: Trajectory to export.
        motor_types: List of Damiao motor type strings (e.g., "DM4340") for
            each joint. Must have length equal to trajectory.n_joints.
        can_ids: List of CAN IDs for each joint. Must have length equal to
            trajectory.n_joints.
        output_path: Path to the output CSV file.

    Returns:
        Dictionary with clamping statistics. Keys are "joint{i}_clamped_pct"
        for each joint i, with values being the percentage of trajectory points
        that were clamped (0.0 to 100.0).

    Raises:
        ValueError: If motor_types or can_ids have incorrect length, or if
            any motor type is not recognized.
        IOError: If the output file cannot be written.
    """
    if len(motor_types) != trajectory.n_joints:
        raise ValueError(
            f"motor_types has length {len(motor_types)}, expected {trajectory.n_joints}"
        )
    if len(can_ids) != trajectory.n_joints:
        raise ValueError(
            f"can_ids has length {len(can_ids)}, expected {trajectory.n_joints}"
        )

    # Validate motor types and get limits
    motor_limits = []
    for motor_type in motor_types:
        if motor_type not in DAMIAO_MOTOR_LIMITS:
            valid_types = ", ".join(sorted(DAMIAO_MOTOR_LIMITS.keys()))
            raise ValueError(
                f"Unknown motor type '{motor_type}'. Valid types: {valid_types}"
            )
        motor_limits.append(DAMIAO_MOTOR_LIMITS[motor_type])

    # Clamp trajectory values and track clamping statistics
    n_samples = len(trajectory.time)
    clamped_position = np.zeros_like(trajectory.position)
    clamped_velocity = np.zeros_like(trajectory.velocity)
    clamped_torque = np.zeros_like(trajectory.position)  # Assume zero torque

    clamping_stats = {}

    for j in range(trajectory.n_joints):
        pos_max, vel_max, torque_max = motor_limits[j]

        # Clamp positions (symmetric limits)
        pos_j = trajectory.position[:, j]
        clamped_pos_j = np.clip(pos_j, -pos_max, pos_max)
        n_pos_clamped = np.sum(pos_j != clamped_pos_j)

        # Clamp velocities (symmetric limits)
        vel_j = trajectory.velocity[:, j]
        clamped_vel_j = np.clip(vel_j, -vel_max, vel_max)
        n_vel_clamped = np.sum(vel_j != clamped_vel_j)

        # Torque is zero, but clamp anyway for consistency
        torque_j = np.zeros(n_samples)
        clamped_torque_j = np.clip(torque_j, -torque_max, torque_max)
        n_torque_clamped = np.sum(torque_j != clamped_torque_j)

        clamped_position[:, j] = clamped_pos_j
        clamped_velocity[:, j] = clamped_vel_j
        clamped_torque[:, j] = clamped_torque_j

        # Compute clamping percentage
        n_clamped = n_pos_clamped + n_vel_clamped + n_torque_clamped
        clamped_pct = 100.0 * n_clamped / (3 * n_samples)
        clamping_stats[f"joint{j}_clamped_pct"] = clamped_pct

    # Write CSV file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        # Write header comments with motor info
        for j in range(trajectory.n_joints):
            f.write(f"# motor_type={motor_types[j]},can_id={can_ids[j]},mode=MIT\n")

        # Write column headers
        header_parts = ["time"]
        for j in range(trajectory.n_joints):
            header_parts.extend([f"j{j}_pos", f"j{j}_vel", f"j{j}_tau"])
        f.write(",".join(header_parts) + "\n")

        # Write data rows
        for i in range(n_samples):
            row_parts = [f"{trajectory.time[i]:.3f}"]
            for j in range(trajectory.n_joints):
                row_parts.append(f"{clamped_position[i, j]:.4f}")
                row_parts.append(f"{clamped_velocity[i, j]:.4f}")
                row_parts.append(f"{clamped_torque[i, j]:.4f}")
            f.write(",".join(row_parts) + "\n")

    return clamping_stats


def _find_body_with_joint(root: ET.Element, joint_name: str) -> ET.Element | None:
    """Find the body element that contains a joint with the given name.

    Recursively searches the XML tree for a body containing a joint with the
    specified name.

    Args:
        root: Root element of the XML tree (or subtree).
        joint_name: Name of the joint to search for.

    Returns:
        The body element containing the joint, or None if not found.
    """
    # Check if this element is a body
    if root.tag == "body":
        # Check if this body contains the joint
        for joint in root.findall("joint"):
            if joint.get("name") == joint_name:
                return root
        
        # Recursively search child bodies
        for child in root.findall("body"):
            result = _find_body_with_joint(child, joint_name)
            if result is not None:
                return result
    else:
        # Not a body element, search all descendant bodies
        for body in root.iter("body"):
            for joint in body.findall("joint"):
                if joint.get("name") == joint_name:
                    return body

    return None


def _find_parent(root: ET.Element, target: ET.Element) -> ET.Element | None:
    """Find the parent element of a target element.

    Args:
        root: Root element of the XML tree.
        target: Target element to find the parent of.

    Returns:
        The parent element, or None if not found.
    """
    for parent in root.iter():
        if target in parent:
            return parent
    return None
