"""
Real-time Cartesian velocity viewer for each joint.
Displays linear velocity (vx, vy, vz) at each joint body in the MuJoCo viewer.

Run with mjpython:
    mjpython velocity_viewer.py
"""
import mujoco
import mujoco.viewer
import numpy as np
import os


def system_id():
    """
    System identification function for the Kinova arm.
    Uses the guess model for parameter estimation.
    """
    from motor import Motor
    
    model_path = os.path.join(os.path.dirname(__file__), "model", "kinova_fullinertia_guess.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Create 7 motor objects with MIT controller
    # PD gains estimated for Kinova arm:
    # - Joints 1-4 (shoulder/elbow): higher gains for larger joints
    # - Joints 5-7 (wrist): lower gains for smaller joints
    # 
    # Friction parameters (Stribeck model) estimated for harmonic drive actuators:
    # - T_coulomb: Coulomb friction ~1-3% of max torque for larger joints, ~2-4% for wrist
    # - T_static: Static friction ~1.2-1.5x Coulomb friction
    # - omega_s: Stribeck velocity ~0.1-0.3 rad/s typical for harmonic drives
    motors = []
    
    # Joint parameters: (position_limit, velocity_limit, torque_limit, kp, kd, T_coulomb, T_static, omega_s)
    joint_params = [
        (np.pi, 2.0, 36.0, 40.0, 12.0, 0.8, 1.2, 0.15),  # Joint 1 - base rotation
        (np.pi, 2.0, 36.0, 50.0, 15.0, 1.0, 1.5, 0.15),  # Joint 2 - shoulder
        (np.pi, 2.0, 36.0, 40.0, 12.0, 0.8, 1.2, 0.15),  # Joint 3 - shoulder rotation
        (np.pi, 2.0, 36.0, 30.0, 10.0, 0.7, 1.0, 0.15),  # Joint 4 - elbow
        (np.pi, 2.5, 9.8, 15.0, 5.0, 0.25, 0.35, 0.2),   # Joint 5 - wrist 1
        (np.pi, 2.5, 9.8, 15.0, 5.0, 0.25, 0.35, 0.2),   # Joint 6 - wrist 2
        (np.pi, 2.5, 9.8, 10.0, 4.0, 0.2, 0.3, 0.2),     # Joint 7 - wrist 3
    ]
    
    for i, (pos_lim, vel_lim, torque_lim, kp, kd, T_c, T_s, omega_s) in enumerate(joint_params):
        motor = Motor(
            position_limit=pos_lim,
            velocity_limit=vel_lim,
            torque_limit=torque_lim,
            use_mit_controller=True,
            T_coulomb=T_c,
            T_static=T_s,
            omega_s=omega_s
        )
        motor.set_mit_params(kp=kp, kd=kd)
        motors.append(motor)
    
    # Generate demo trajectory - test one joint at a time (0-6), or None for all
    TEST_JOINT = None  # Change this to test different joints: 0-6, or None for all
    trajectory, velocity = generate_demo_trajectory(seconds=10.0, test_joint=TEST_JOINT)
    
    # Initialize simulation at first trajectory position
    data.qpos[:7] = trajectory[0, :]
    data.qvel[:7] = velocity[0, :]
    mujoco.mj_forward(model, data)
    
    # Initialize motors with current state
    for i, motor in enumerate(motors):
        motor.update(data.qpos[i], data.qvel[i])
    
    # Replay motion with feedforward + feedback control
    frame_idx = 0
    num_frames = trajectory.shape[0]
    
    dt = model.opt.timestep  # Use model timestep
    slowdown = 1  # Slow down playback for visualization
    
    import time
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if frame_idx >= num_frames:
                # Loop trajectory
                frame_idx = 0
                data.qpos[:7] = trajectory[0, :]
                data.qvel[:7] = velocity[0, :]
                mujoco.mj_forward(model, data)
                for i, motor in enumerate(motors):
                    motor.reset()
                    motor.update(data.qpos[i], data.qvel[i])
            
            # Step 1: Calculate feedforward torques using mj_inverse
            # Set qacc to desired acceleration (from trajectory or zero for position tracking)
            if frame_idx < num_frames - 1:
                # Compute desired acceleration from trajectory velocity difference
                desired_acc = (velocity[frame_idx + 1, :] - velocity[frame_idx, :]) / dt
            else:
                desired_acc = np.zeros(7)
            
            data.qacc[:7] = desired_acc
            
            # Save current velocities, zero them for mj_inverse, then restore
            saved_qvel = data.qvel[:7].copy()
            data.qvel[:7] = 0.0
            
            # Use mj_inverse to compute required torques for desired motion
            mujoco.mj_inverse(model, data)
            feedforward_torques = data.qfrc_inverse[:7].copy()
            
            # Restore velocities
            data.qvel[:7] = saved_qvel
            
            # Step 2: For each motor, set feedforward torque and desired position from trajectory
            for i, motor in enumerate(motors):
                # For non-test joints, hold at zero with high gains
                if TEST_JOINT is not None and i != TEST_JOINT:
                    motor.set_mit_params(
                        kp=200.0,  # High stiffness to lock joint
                        kd=20.0,
                        desired_pos=0.0,
                        desired_vel=0.0,
                        ff_torque=feedforward_torques[i]
                    )
                else:
                    motor.set_mit_params(
                        desired_pos=trajectory[frame_idx, i],
                        desired_vel=velocity[frame_idx, i],
                        ff_torque=feedforward_torques[i]
                    )
            
            # Step 3: Compute motor output torques (includes PD feedback + feedforward)
            output_torques = np.zeros(7)
            for i, motor in enumerate(motors):
                output_torques[i] = motor.get_output_torque()
            
            # Step 4: Apply torques to simulation and step
            data.ctrl[:7] = output_torques
            mujoco.mj_step(model, data)
            
            # Step 5: Update motor states with new simulation state
            for i, motor in enumerate(motors):
                motor.update(data.qpos[i], data.qvel[i])
            
            frame_idx += 1
            
            # Slow down playback
            time.sleep(dt * slowdown)
            
            viewer.sync()

def get_cartesian_velocities(model, data):
    """Compute Cartesian velocity at each joint site in local body frame."""
    velocities = []
    
    for name, site_id, body_id in joint_info:
        # Allocate Jacobians
        jacp = np.zeros((3, model.nv))  # translational
        jacr = np.zeros((3, model.nv))  # rotational
        
        # Compute Jacobian for this site (at joint origin)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        
        # Cartesian velocity in world frame = J @ qvel
        linear_vel_world = jacp @ data.qvel
        angular_vel_world = jacr @ data.qvel
        
        # Get body rotation matrix (3x3) from data.xmat (stored as 9-element flat array)
        R_body = data.xmat[body_id].reshape(3, 3)
        
        # Transform to local body frame: v_local = R^T @ v_world
        linear_vel_local = R_body.T @ linear_vel_world
        angular_vel_local = R_body.T @ angular_vel_world
        
        velocities.append({
            'name': name,
            'site_id': site_id,
            'body_id': body_id,
            'linear_world': linear_vel_world.copy(),
            'linear_local': linear_vel_local.copy(),
            'angular_world': angular_vel_world.copy(),
            'angular_local': angular_vel_local.copy(),
            'speed': np.linalg.norm(linear_vel_local)
        })
    
    return velocities

def format_velocity_display(velocities):
    """Format velocities for display."""
    lines = ["Cartesian Velocities in Local Frame (m/s):", ""]
    for v in velocities:
        lv = v['linear_local']
        lines.append(f"{v['name']:>8}: [{lv[0]:+6.3f}, {lv[1]:+6.3f}, {lv[2]:+6.3f}]  |v|={v['speed']:.3f}")
    return "\n".join(lines)

# Generate a simple oscillating trajectory for demo
def generate_demo_trajectory(seconds: float, hz: int = 1000, test_joint: int = None):
    """Generate oscillating motion on joints.
    
    Args:
        seconds: Duration of trajectory
        hz: Sample rate
        test_joint: If specified (0-6), only move this joint. If None, move all.
    
    Returns:
        trajectory: (num_samples, 7) array of joint positions
        velocity: (num_samples, 7) array of joint velocities
    """
    num_samples = int(seconds * hz)
    t = np.linspace(0, seconds, num_samples)
    trajectory = np.zeros((num_samples, 7))
    velocity = np.zeros((num_samples, 7))
    
    # Different frequencies for each joint
    freqs = [0.3, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3]
    amps = [0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2]
    
    if test_joint is not None:
        # Only move the specified joint
        joints_to_move = [test_joint]
        print(f"Testing joint {test_joint} only")
    else:
        # Move first 4 joints
        joints_to_move = range(4)
    
    for j in joints_to_move:
        omega = 2 * np.pi * freqs[j]
        trajectory[:, j] = amps[j] * np.sin(omega * t)
        velocity[:, j] = amps[j] * omega * np.cos(omega * t)
    
    return trajectory, velocity

if __name__ == "__main__":
    system_id()
