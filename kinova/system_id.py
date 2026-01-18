"""
Simple MuJoCo viewer for inspecting joint positions with gravity compensation.
Uses mjpython's built-in interactive viewer - drag the arm to see joint values.
Now uses Motor class with 2ms delay for torque commands.

Run with mjpython:
    mjpython joint_viewer.py
"""
import mujoco
import mujoco.viewer
import numpy as np
import os
from motor import Motor
from ik_solver import KinovaIKSolver

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "kinova.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Create 7 motor objects (one for each joint)
# Using direct torque command mode (not MIT controller)
# Stribeck friction parameters estimated for Kinova arm joints
friction_params = [
    # J1, J2 (base/shoulder) - larger joints
    {"T_coulomb": 0.4, "T_static": 0.5, "omega_s": 0.05, "delta": 2},
    {"T_coulomb": 0.4, "T_static": 0.5, "omega_s": 0.05, "delta": 2},
    # J3, J4 (elbow) - mid joints
    {"T_coulomb": 0.25, "T_static": 0.35, "omega_s": 0.05, "delta": 2},
    {"T_coulomb": 0.25, "T_static": 0.35, "omega_s": 0.05, "delta": 2},
    # J5, J6, J7 (wrist) - smaller joints
    {"T_coulomb": 0.15, "T_static": 0.2, "omega_s": 0.05, "delta": 2},
    {"T_coulomb": 0.15, "T_static": 0.2, "omega_s": 0.05, "delta": 2},
    {"T_coulomb": 0.15, "T_static": 0.2, "omega_s": 0.05, "delta": 2},
]

motors = [
    Motor(position_limit=2*np.pi, velocity_limit=10.0, torque_limit=50.0,
          use_mit_controller=False, **params)
    for params in friction_params
]

# Set initial joint positions (modify these to change starting pose)
initial_qpos = [0, 0, 0, 0, 0, 0, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

print("=" * 50)
print("Kinova Joint Viewer (with Motor class + 2ms delay)")
print("=" * 50)
print("Drag the arm in the viewer to move joints.")
print("Joint positions are printed continuously.")
print("Torque commands pass through Motor objects with 2ms delay.")
print("Close the viewer window to quit.")
print("=" * 50)

def generate_trajectory(joint_number: int, seconds: float) -> np.ndarray:
    """
    Generate a trajectory for system identification.
    
    Args:
        joint_number: Which joint to generate motion for (1-7). 
                      Joints above this number are set to 0.
        seconds: Duration of the trajectory in seconds.
    
    Returns:
        np.ndarray: Array of shape (seconds * 1000, 7) containing joint positions
                    at 1000Hz. Columns represent joints 1-7.
    
    Joint 1 frequency: 1

    Joint 2 frequency: 1

    Joint 3 frequency: 1

    Joint 4 frequency: 1

    Joint 5 frequency: 1

    Joint 6 frequency: 1

    Joint 7 frequency: 1
    """
    num_samples = int(seconds * 1000)
    frequencies = [0.3, 0.47, 0.53, 0.59, 0.73, 0, 0]  # Hz for joints 1-7
    time_array = np.linspace(0, seconds, num_samples)
    
    trajectory = np.zeros((num_samples, 7))
    
    # Generate random amplitudes and phase offsets for joints up to joint_number
    amplitudes = np.random.uniform(np.pi / 4, np.pi / 2, joint_number)
    phase_offsets = np.random.uniform(0, 2 * np.pi, joint_number)
    
    # Generate trajectory for each joint
    for j in range(joint_number):
        trajectory[:, j] = amplitudes[j] * np.sin(
            2 * np.pi * frequencies[j] * time_array + phase_offsets[j]
        )
    # Joints above joint_number remain at 0 (already initialized)
    
    return trajectory


def print_joint_positions(data):
    """Print current joint positions in a readable format."""
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
    positions = data.qpos[:7]
    
    # Clear line and print
    print("\rJoints: " + " | ".join(
        f"{name}: {pos:+.3f}" for name, pos in zip(joint_names, positions)
    ), end="", flush=True)

def apply_gravity_compensation(model, data):
    """Compute gravity compensation torques and send through motor objects."""
    # Save state that we need to restore
    xfrc_applied_save = data.xfrc_applied.copy()
    qvel_save = data.qvel.copy()
    
    # Zero out external forces and velocities for inverse dynamics calculation
    data.xfrc_applied[:] = 0
    data.qvel[:] = 0
    data.qacc[:] = 0  # Zero acceleration
    
    # Compute gravity compensation torques
    mujoco.mj_inverse(model, data)
    gravity_torques = data.qfrc_inverse[:7].copy()
    
    # Restore external forces and velocities
    data.xfrc_applied[:] = xfrc_applied_save
    data.qvel[:] = qvel_save
    
    # Send torque commands through motor objects
    for i, motor in enumerate(motors):
        motor.set_torque_command(gravity_torques[i])
        motor.update(data.qpos[i], data.qvel[i])
        data.ctrl[i] = motor.get_output_torque()

if __name__ == "__main__":
    print("Generating 5-second trajectory...")
    trajectory = generate_trajectory(joint_number=4, seconds=5.0)
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Trajectory playback at 1000Hz (1ms per frame)
    frame_idx = 0
    num_frames = trajectory.shape[0]
    
    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Set joint positions from trajectory
            if frame_idx < num_frames:
                data.qpos[:7] = trajectory[frame_idx, :]
                frame_idx += 1
            
            # Update kinematics without physics simulation
            mujoco.mj_forward(model, data)
            
            # Print joint positions
            print_joint_positions(data)
            
            # Sync viewer
            viewer.sync()

    print("\n\nFinal joint positions:")
    print(f"qpos = {list(np.round(data.qpos[:7], 4))}")
