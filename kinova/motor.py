import numpy as np


class Motor:
    """
    Motor class implementing MIT controller 14-bit encoder quantization.
    Each method call represents 1ms time step.
    """
    
    def __init__(self, position_limit=2*np.pi, velocity_limit=10.0, torque_limit=10.0, use_mit_controller=True,
                 T_coulomb=None, T_static=None, omega_s=None, delta=2):
        """
        Initialize motor with MIT controller parameters.
        
        Args:
            position_limit: Maximum position range (radians)
            velocity_limit: Maximum velocity (rad/s)
            torque_limit: Maximum torque (Nm)
            use_mit_controller: If True, use MIT controller. If False, use direct torque commands.
            T_coulomb: Coulomb friction torque (Nm)
            T_static: Static friction torque (Nm)
            omega_s: Stribeck velocity (rad/s)
            delta: Shape factor for Stribeck model
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
        
        # 14-bit encoder (0 to 16383 counts)
        self.encoder_bits = 14
        self.encoder_resolution = 2**self.encoder_bits  # 16384 counts
        self.dt = 0.001  # 1ms time step
        
        # Stribeck friction model parameters
        self.T_coulomb = T_coulomb  # Coulomb friction torque (Nm)
        self.T_static = T_static  # Static friction torque (Nm)
        self.omega_s = omega_s  # Stribeck velocity (rad/s)
        self.delta = delta  # Shape factor
        
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
    
    def compute_torque(self, debug=False):
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
            
            if debug:
                print(f"  PD debug: des_pos={self.desired_position:.4f}, act_pos={self.actual_position:.4f}, pos_err={position_error:.4f}")
                print(f"            des_vel={self.desired_velocity:.4f}, act_vel={self.actual_velocity:.4f}, vel_err={velocity_error:.4f}")
                print(f"            kp={self.kp}, kd={self.kd}, ff={self.feedforward_torque:.4f}")
                print(f"            kp*pos_err={self.kp * position_error:.4f}, kd*vel_err={self.kd * velocity_error:.4f}")
                print(f"            torque_cmd={torque_cmd:.4f}")
        else:
            # Direct torque command
            torque_cmd = self.commanded_torque
        
        # Apply torque limits
        torque_cmd = np.clip(torque_cmd, -self.torque_limit, self.torque_limit)
        
        return torque_cmd
    
    def update(self, true_position, true_velocity, initialize=False):
        """
        Update motor state (1ms time step).
        
        Args:
            true_position: True position from simulation (rad)
            true_velocity: True velocity from simulation (rad/s)
            initialize: If True, initialize prev_encoder_position to current position
        """
        # Store true state from simulation
        self.true_position = true_position
        self.true_velocity = true_velocity
        
        # Get quantized encoder position
        self.actual_position = self.get_encoder_position()
        
        if initialize:
            # On first call, set prev position to current to avoid velocity spike
            self.prev_encoder_position = self.actual_position
            self.actual_velocity = 0.0
        else:
            # Estimate velocity from encoder position difference
            self.actual_velocity = self.get_encoder_velocity()
        
        # Note: torque is computed in get_output_torque() after set_mit_params() is called
    
    def get_output_torque(self, Bv=0.0, debug=False):
        """
        Compute and return torque output with Stribeck friction model applied.
        
        Args:
            Bv: Viscous friction coefficient (Nm·s/rad)
            debug: Print debug info
        
        Returns:
            Torque output with friction (Nm)
        """
        # Compute torque command using current state and desired setpoints
        torque_cmd = self.compute_torque(debug=debug)
        
        # Use cached encoder velocity (computed in update())
        omega = self.actual_velocity
        
        # Apply Stribeck friction model if parameters are set
        if all(param is not None for param in [self.T_coulomb, self.T_static, self.omega_s]):
            # T_f = [T_c + (T_s - T_c) * e^(-(|omega/omega_s|)^delta)] * sgn(omega) + B_v * omega
            stribeck_friction = (
                (self.T_coulomb + (self.T_static - self.T_coulomb) * 
                 np.exp(-np.abs(omega / self.omega_s) ** self.delta)) * 
                np.sign(omega)
            )
            
            friction_torque = stribeck_friction + Bv * omega
            
            # Apply friction to output torque
            output_torque = torque_cmd - friction_torque
        else:
            output_torque = torque_cmd
        
        return output_torque
    
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
        Note: This should only be called once per timestep in update().
        
        Returns:
            Estimated velocity (rad/s)
        """
        current_encoder_pos = self.get_encoder_position()
        
        # Velocity = (current_pos - prev_pos) / dt
        velocity = (current_encoder_pos - self.prev_encoder_position) / self.dt
        
        # Update previous position for next iteration
        self.prev_encoder_position = current_encoder_pos
        
        return velocity
    
    def get_cached_encoder_velocity(self):
        """
        Get the cached encoder velocity (computed in update()).
        Use this instead of get_encoder_velocity() to avoid side effects.
        """
        return self.actual_velocity
    
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
