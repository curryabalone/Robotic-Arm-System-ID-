import numpy as np
class DMMotor4340:
    def __init__(self, viscious_friction = 0.134, static_friction = 0.6):
        self.viscious_friction = viscious_friction
        self.static_friction = static_friction
        self.command_frame = [[0,0,0,0,0,0,0] for _ in range(3)]
        self.pos = [0,0,0]
        self.vel = [0,0,0]
        self.pointer = 2
        self.sensor_pointer = 2
        self.noise_std = 0.1  # Standard deviation of the torque noise (Nm) turned off because it ruins the simulation
    def update(self, pos, vel):
        self.pos[self.sensor_pointer] = pos
        self.vel[self.sensor_pointer] = vel
        self.sensor_pointer = (self.sensor_pointer + 1) % 3
    def mitcontrol(self, kp, kd, q, dq, tau):
        self.command_frame[self.pointer] = [kp, kd, q, dq, tau]
        self.pointer = (self.pointer + 1) % 3
    def output_torque(self):
        current_frame = self.command_frame[(self.pointer-2) % 3]
        current_pos = self.pos[(self.sensor_pointer-2) % 3]
        current_vel = self.vel[(self.sensor_pointer-2) % 3]
        latest_vel = self.vel[(self.sensor_pointer-1) % 3] # Instant velocity for physical friction
        kp = current_frame[0]
        kd = current_frame[1]
        q = current_frame[2]
        dq = current_frame[3]
        tau = current_frame[4]
        expected_torque = kp * (q - current_pos) + kd * (dq - current_vel) + tau
        noise = np.random.normal(0, self.noise_std, 1)
        return expected_torque - self.viscious_friction * latest_vel + noise[0]
    

        
