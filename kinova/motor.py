import numpy as np
from collections import deque


class Motor:
    """
    Motor class implementing MIT controller with 2ms delay and 16-bit encoder quantization.
    Each method call represents 1ms time step.
    """
    
    def __init__(self, position_limit=2*np.pi, velocity_limit=10.0, torque_limit=10.0, use_mit_controller=True):
        """
        Initialize motor with MIT controller parameters.
        
        Args:
            position_limit: Maximum position range (radians)
            velocity_limit: Maximum velocity (rad/s)
            torque_limit: Maximum torque (Nm)
            use_mit_controller: If True, use MIT controller. If False, use direct torque commands.
        """
        # Control mode
        self.use_mit_controller = use_mit_controller
        
        # MIT Controller parameters
        self.kp = 0.0  # Position gain
        self.kd = 0.0  # Velocity gain
        self.desired_position = 0.0  # Desired position (rad)
        self.desired_velocity = 0.0  # Desired velocity (rad/s)
        self.feedforward_torque = 0.0  # Feedforward torque (Nm)
        
        # Direct torque command (for non-MIT mode)
        self.commanded_torque = 0.0  # Direct torque command (Nm)
        
        # Motor state (ground truth from simulation)
        self.true_position = 0.0  # True position from simulation (rad)
        self.true_velocity = 0.0  # True velocity from simulation (rad/s)
        
        # Estimated state (from encoder)
        self.actual_position = 0.0  # Estimated position from encoder (rad)
        self.actual_velocity = 0.0  # Estimated velocity from encoder (rad/s)
        self.actual_torque = 0.0  # Actual torque output (Nm)
        self.prev_encoder_position = 0.0  # Previous encoder position for velocity estimation
        
        # Limits
        self.position_limit = position_limit
        self.velocity_limit = velocity_limit
        self.torque_limit = torque_limit
        
        # 2ms delay buffer (3 elements: 2ms ago, 1ms ago, current)
        self.torque_buffer = deque([0.0, 0.0, 0.0], maxlen=3)
        
        # 16-bit encoder (0 to 65535 counts)
        self.encoder_bits = 16
        self.encoder_resolution = 2**self.encoder_bits  # 65536 counts
        self.dt = 0.001  # 1ms time step
        
    def set_mit_params(self, kp=None, kd=None, desired_pos=None, desired_vel=None, ff_torque=None):
        """
        Set MIT controller parameters.
        
        Args:
            kp: Position gain
            kd: Velocity gain
            desired_pos: Desired position (rad)
            desired_vel: Desired velocity (rad/s)
            ff_torque: Feedforward torque (Nm)
        """
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd
        if desired_pos is not None:
            self.desired_position = desired_pos
        if desired_vel is not None:
            self.desired_velocity = desired_vel
        if ff_torque is not None:
            self.feedforward_torque = ff_torque
    
    def set_torque_command(self, torque):
        """
        Set direct torque command (for non-MIT mode).
        
        Args:
            torque: Commanded torque (Nm)
        """
        self.commanded_torque = torque
    
    def compute_torque(self):
        """
        Compute commanded torque using MIT controller or direct command.
        
        Returns:
            Commanded torque (Nm)
        """
        if self.use_mit_controller:
            # MIT controller: tau = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
            position_error = self.desired_position - self.actual_position
            velocity_error = self.desired_velocity - self.actual_velocity
            
            torque_cmd = (self.kp * position_error + 
                         self.kd * velocity_error + 
                         self.feedforward_torque)
        else:
            # Direct torque command
            torque_cmd = self.commanded_torque
        
        # Apply torque limits
        torque_cmd = np.clip(torque_cmd, -self.torque_limit, self.torque_limit)
        
        return torque_cmd
    
    def update(self, true_position, true_velocity):
        """
        Update motor state (1ms time step).
        
        Args:
            true_position: True position from simulation (rad)
            true_velocity: True velocity from simulation (rad/s)
        """
        # Store true state from simulation
        self.true_position = true_position
        self.true_velocity = true_velocity
        
        # Get quantized encoder readings
        self.actual_position = self.get_encoder_position()
        self.actual_velocity = self.get_encoder_velocity()
        
        # Compute new torque command using estimated state
        torque_cmd = self.compute_torque()
        
        # Add to delay buffer
        self.torque_buffer.append(torque_cmd)
        
        # Output is the delayed torque (2ms ago)
        self.actual_torque = self.torque_buffer[0]
    
    def get_output_torque(self):
        """
        Get the delayed torque output.
        
        Returns:
            Torque output with 2ms delay (Nm)
        """
        return self.actual_torque
    
    def get_encoder_position(self):
        """
        Get quantized position from 16-bit encoder.
        
        Returns:
            Quantized position (rad)
        """
        # Map true position to encoder counts
        # Assuming symmetric range: [-position_limit, +position_limit]
        normalized_pos = (self.true_position + self.position_limit) / (2 * self.position_limit)
        encoder_count = int(normalized_pos * (self.encoder_resolution - 1))
        
        # Clip to valid range
        encoder_count = np.clip(encoder_count, 0, self.encoder_resolution - 1)
        
        # Convert back to position (quantized)
        quantized_pos = (encoder_count / (self.encoder_resolution - 1)) * (2 * self.position_limit) - self.position_limit
        
        return quantized_pos
    
    def get_encoder_velocity(self):
        """
        Estimate velocity from encoder position difference.
        
        Returns:
            Estimated velocity (rad/s)
        """
        current_encoder_pos = self.get_encoder_position()
        
        # Velocity = (current_pos - prev_pos) / dt
        velocity = (current_encoder_pos - self.prev_encoder_position) / self.dt
        
        # Update previous position for next iteration
        self.prev_encoder_position = current_encoder_pos
        
        return velocity
    
    def get_encoder_count(self):
        """
        Get raw encoder count (0 to 65535).
        
        Returns:
            Encoder count (int)
        """
        normalized_pos = (self.true_position + self.position_limit) / (2 * self.position_limit)
        encoder_count = int(normalized_pos * (self.encoder_resolution - 1))
        encoder_count = np.clip(encoder_count, 0, self.encoder_resolution - 1)
        
        return encoder_count
    
    def reset(self):
        """Reset motor to initial state."""
        self.true_position = 0.0
        self.true_velocity = 0.0
        self.actual_position = 0.0
        self.actual_velocity = 0.0
        self.actual_torque = 0.0
        self.prev_encoder_position = 0.0
        self.desired_position = 0.0
        self.desired_velocity = 0.0
        self.feedforward_torque = 0.0
        self.torque_buffer = deque([0.0, 0.0, 0.0], maxlen=3)
