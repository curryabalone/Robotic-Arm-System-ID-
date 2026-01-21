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

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "kinova_fullinertia.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Find joint site IDs and corresponding body IDs (for rotation matrices)
joint_site_names = [f"joint{i}" for i in range(1, 8)]
joint_info = []
for name in joint_site_names:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    # Get the body that contains this site
    body_id = model.site_bodyid[site_id] if site_id >= 0 else -1
    if site_id >= 0:
        joint_info.append((name, site_id, body_id))

print(f"Tracking {len(joint_info)} joints: {[name for name, _, _ in joint_info]}")

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
def generate_demo_trajectory(seconds: float, hz: int = 1000):
    """Generate oscillating motion on first 4 joints."""
    num_samples = int(seconds * hz)
    t = np.linspace(0, seconds, num_samples)
    trajectory = np.zeros((num_samples, 7))
    
    # Different frequencies for each joint
    freqs = [0.3, 0.4, 0.5, 0.6]
    amps = [0.5, 0.4, 0.3, 0.3]
    
    for j in range(4):
        trajectory[:, j] = amps[j] * np.sin(2 * np.pi * freqs[j] * t)
    
    return trajectory

if __name__ == "__main__":
    print("=" * 60)
    print("Kinova Cartesian Velocity Viewer")
    print("=" * 60)
    print("Shows real-time Cartesian velocity at each joint body.")
    print("Drag the arm or watch the trajectory playback.")
    print("=" * 60)
    
    # Generate trajectory
    trajectory = generate_demo_trajectory(seconds=10.0)
    frame_idx = 0
    num_frames = trajectory.shape[0]
    prev_qpos = trajectory[0, :].copy()
    
    # For numerical velocity estimation
    dt = 0.001  # 1ms timestep
    slowdown = 5  # Slow down playback by 5x for visual verification
    
    import time
    last_print = 0
    print_interval = 0.1  # Print every 100ms
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Update position from trajectory
            if frame_idx < num_frames:
                data.qpos[:7] = trajectory[frame_idx, :]
                # Estimate velocity numerically
                data.qvel[:7] = (trajectory[frame_idx, :] - prev_qpos) / dt
                prev_qpos = trajectory[frame_idx, :].copy()
                frame_idx += 1
            else:
                # Loop trajectory
                frame_idx = 0
                prev_qpos = trajectory[0, :].copy()
            
            mujoco.mj_forward(model, data)
            
            # Compute Cartesian velocities
            velocities = get_cartesian_velocities(model, data)
            
            # Print formatted table at intervals
            now = time.time()
            if now - last_print > print_interval:
                last_print = now
                # Clear screen and print table
                print("\033[2J\033[H")  # Clear screen, cursor to top
                print("=" * 80)
                print("  CARTESIAN VELOCITIES IN LOCAL BODY FRAME (m/s)")
                print("=" * 80)
                print(f"{'Joint':<10} {'vx_local':>12} {'vy_local':>12} {'vz_local':>12} {'|v|':>10}")
                print("-" * 80)
                for v in velocities:
                    lv = v['linear_local']
                    print(f"{v['name']:<10} {lv[0]:>+12.4f} {lv[1]:>+12.4f} {lv[2]:>+12.4f} {v['speed']:>10.4f}")
                print("-" * 80)
                print(f"Frame: {frame_idx}/{num_frames} (slowdown: {slowdown}x)")
            
            # Slow down playback
            time.sleep(dt * slowdown)
            
            viewer.sync()
    
    print("\n\nDone!")
