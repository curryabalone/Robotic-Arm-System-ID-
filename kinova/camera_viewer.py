"""
Kinova arm simulation with wrist camera viewer.
Saves camera frames to file for viewing.
"""
import mujoco
import mujoco.viewer
import numpy as np
import os
import subprocess
import time

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "kinova.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Create renderer for wrist camera
cam_width, cam_height = 640, 480
renderer = mujoco.Renderer(model, height=cam_height, width=cam_width)

# Set initial joint positions
data.qpos[:] = [0, 0, 0, 0, 0, 0, 0]
mujoco.mj_forward(model, data)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Simulation running.")
print("Press 'S' in viewer to save current wrist camera snapshot.")
print("Close window to quit.")

frame_count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Save camera snapshot every 500 steps (~0.5s at 1kHz)
        frame_count += 1
        if frame_count % 500 == 0:
            renderer.update_scene(data, camera="wrist_cam")
            frame = renderer.render()
            
            plt.figure(figsize=(8, 6))
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f'Wrist Camera - Frame {frame_count}')
            plt.savefig('wrist_cam.png', bbox_inches='tight')
            plt.close()
            print(f"Saved wrist_cam.png (frame {frame_count})")
        
        viewer.sync()
        
        # Realtime sync
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

renderer.close()
print("Done. Opening last camera frame...")
subprocess.run(['open', 'wrist_cam.png'])
