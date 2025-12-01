import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent crashes
import matplotlib.pyplot as plt
import subprocess
from dm_motor_4340 import DMMotor4340 
# Path to the XML file
xml_path = "./4DOF_Arm.xml"

def main():
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize motor
    motors = [DMMotor4340() for _ in range(4)]
    dict = {"motor1": motors[0], "motor2": motors[1], "motor3": motors[2], "motor4": motors[3]}
    kd_dict = [0.25, 0.1, 0.1, 0]    
    # Prime the motors with initial state to prevent "limp" start
    # 1. Compute initial gravity compensation
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    initial_torques = data.qfrc_inverse.copy()
    
    # 2. Fill motor buffers (delay is 3 steps, so we send 3 commands)
    print("Priming motors...")
    for _ in range(3):
        for key, motor in enumerate(motors):
            # Send initial hold command: target_vel=0, target_torque=gravity_comp
            motor.mitcontrol(0, kd_dict[key], data.qpos[key], 0.0, initial_torques[key])
            # Also update sensor state so the motor doesn't think it jumped from 0
            motor.update(data.qpos[key], 0.0)

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable perturbation visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
        
        print("Simulation started.")
        print("Use Ctrl + Left Click and Drag to apply force (drag the arm).")
        print("Use Ctrl + Right Click and Drag to apply torque.")
        
        # Data recording buffers
        time_data = []
        motor_torque_data = []
        applied_torque_data = []
        pos_data = []

        start_time = time.time()
        try:
            while viewer.is_running():
                step_start = time.time()

                # 1. Compute Gravity Compensation Torques
                # We want to hold the current position against gravity, but allow external forces (like mouse drag)
                # to move the robot. So we must NOT compensate for xfrc_applied.
                
                # Lock the viewer while accessing/modifying data that is shared
                with viewer.lock():
                    # Save current state
                    qpos_current = data.qpos.copy()
                    qvel_current = data.qvel.copy()
                    qacc_current = data.qacc.copy()
                    # Save external forces (e.g. from mouse) so we can restore them
                    # Multiply by 5 to make mouse interaction stronger as requested 
                    xfrc_applied_current = data.xfrc_applied.copy() * 5.0 
                    qfrc_applied_current = data.qfrc_applied.copy() * 5.0
                    # Update motor objects
                    for key, motor in enumerate(motors):
                        motor.update(qpos_current[key], qvel_current[key])  
                    # Zero out velocities, accelerations, AND external forces for gravity comp calculation
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    data.xfrc_applied[:] = 0.0
                    data.qfrc_applied[:] = 0.0
                    # Compute inverse dynamics (results in data.qfrc_inverse)
                    # This calculates torque needed to hold qpos static against ONLY gravity/coriolis (which is 0).
                    mujoco.mj_inverse(model, data)
                    target_torques = data.qfrc_inverse.copy()
                    # Apply torques to motors
                    for key, motor in enumerate(motors):
                        motor.mitcontrol(10, kd_dict[key], 0, 0.0, target_torques[key])
                    # Restore actual simulation state
                    data.qvel[:] = qvel_current
                    data.qacc[:] = qacc_current
                    data.xfrc_applied[:] = xfrc_applied_current
                    data.qfrc_applied[:] = qfrc_applied_current 
                    # Apply torques to motors
                    for key, motor in enumerate(motors):
                        data.ctrl[key] = motor.output_torque()
                    
                    # Record data for Joint 2 (index 1)
                    current_time = time.time() - start_time
                    time_data.append(current_time)
                    motor_torque_data.append(data.ctrl[1])
                    applied_torque_data.append(data.qfrc_applied[1])
                    pos_data.append(data.qpos[1])                    
                    # 2. Step Simulation
                    mujoco.mj_step(model, data)
                # 3. Sync Viewer
                viewer.sync()

                # Time keeping
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")

    # Plotting after simulation ends
    print(f"Simulation finished. Recorded {len(time_data)} data points.")
    print("Generating plot...")
    
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Torques on Left Axis
        ax1.plot(time_data, motor_torque_data, label='Motor Torque (Joint 2)', color='tab:blue', alpha=0.7)
        ax1.plot(time_data, applied_torque_data, label='Applied Torque (Joint 2)', color='tab:orange', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Torque (Nm)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)

        # Plot Position on Right Axis
        ax2 = ax1.twinx()
        ax2.plot(time_data, pos_data, label='Position (Joint 2)', color='tab:green', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Position (rad)', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('Joint 2 Torque & Position Analysis')
        
        # Save to file
        plt.savefig('torque_plot.png')
        print("Plot saved to 'torque_plot.png'")
        
        print("Opening plot...")
        subprocess.run(['open', 'torque_plot.png'])
        print("Done.")
    except Exception as e:
        print(f"Error during plotting: {e}")
if __name__ == "__main__":
    main()
