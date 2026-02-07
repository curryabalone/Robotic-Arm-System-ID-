"""
Regressor matrix computation for robot system identification.

This module derives and evaluates the spatial dynamics regressor matrix Y such that
the joint torque τ = Y × θ, where θ is the vector of inertial parameters.

The regressor is derived symbolically using SymPy and cached for efficient reuse.
"""

import logging

import sympy as sp
import numpy as np

logger = logging.getLogger(__name__)


def derive_regressor_from_spatial_eqn():
    """
    Derive the symbolic 6x10 regressor matrix for a single rigid body.
    Returns a lambdified function that takes kinematic values and returns the regressor.
    
    The regressor Y satisfies: F_a = Y @ theta
    where F_a is the 6D spatial wrench [torque; force] and theta is the 10 inertial parameters:
    [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    
    Returns:
        regressor_func: callable that takes (wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz)
                        and returns a 6x10 numpy array
    """
    # 1. Define Kinematic Variables (Knowns)
    # --------------------------------------
    # Angular velocity (w), Linear velocity (v)
    wx, wy, wz = sp.symbols('wx wy wz', real=True)
    vx, vy, vz = sp.symbols('vx vy vz', real=True)
    
    # Angular accel (dw), Linear accel (dv)
    dwx, dwy, dwz = sp.symbols('dwx dwy dwz', real=True)
    dvx, dvy, dvz = sp.symbols('dvx dvy dvz', real=True)

    # 2. Define Inertial Parameters (Unknowns to Solve For)
    # -----------------------------------------------------
    # Mass (m)
    m = sp.symbols('m', real=True)
    # Center of Mass * Mass (h = m*c)
    hx, hy, hz = sp.symbols('hx hy hz', real=True)
    # Inertia Tensor (I) - Symmetric 3x3
    Ixx, Iyy, Izz = sp.symbols('Ixx Iyy Izz', real=True)
    Ixy, Ixz, Iyz = sp.symbols('Ixy Ixz Iyz', real=True)

    # The Parameter Vector (Theta)
    theta = sp.Matrix([m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

    # 3. Construct Spatial Vectors (6x1)
    # ----------------------------------
    V_a = sp.Matrix([wx, wy, wz, vx, vy, vz])      # Spatial Velocity
    dV_a = sp.Matrix([dwx, dwy, dwz, dvx, dvy, dvz]) # Spatial Accel

    # 4. Construct Helper Matrices (3x3)
    # ----------------------------------
    # Skew-symmetric operator [x]
    def skew(x, y, z):
        return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    w_skew = skew(wx, wy, wz)
    v_skew = skew(vx, vy, vz)

    # 5. Construct The Big Matrices from Your Image
    # ---------------------------------------------
    # Adjoint Transpose: [ad_Va]^T
    # Top-left: -[w], Top-right: -[v], Bottom-right: -[w]
    ad_T = sp.zeros(6, 6)
    ad_T[0:3, 0:3] = -w_skew
    ad_T[0:3, 3:6] = -v_skew
    ad_T[3:6, 3:6] = -w_skew

    # Spatial Inertia Matrix: G_a
    # Top-left: Inertia (I_bar), Top-right: Skew(h)
    # Bottom-left: Skew(h)^T = -Skew(h), Bottom-right: m * Identity
    
    I_bar = sp.Matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    h_skew = skew(hx, hy, hz)
    
    G_a = sp.zeros(6, 6)
    G_a[0:3, 0:3] = I_bar
    G_a[0:3, 3:6] = h_skew
    G_a[3:6, 0:3] = h_skew.T 
    G_a[3:6, 3:6] = m * sp.eye(3)

    # F_a = G_a * dV_a - ad_T * (G_a * V_a)
    # Note: SymPy handles the matrix multiplication automatically here.
    F_a = G_a * dV_a - ad_T * (G_a * V_a)

    # 7. THE MAGIC: Extract the Regressor
    # -----------------------------------
    # Since F_a is linear in theta, Jacobian returns the coefficient matrix Y.
    Y_regressor = F_a.jacobian(theta)

    # Convert to callable function
    # Order: wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz
    kinematic_vars = (wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz)
    regressor_func = sp.lambdify(kinematic_vars, Y_regressor, modules='numpy')

    return regressor_func


def get_joint_regressor_row(regressor_6x10, joint_axis):
    """
    Extract the row of the regressor corresponding to the joint torque.
    
    The full regressor gives [torque_xyz; force_xyz] = Y @ theta.
    The joint torque is the projection of the angular part onto the joint axis:
        tau_joint = joint_axis @ torque_xyz = joint_axis @ Y[0:3, :] @ theta
    
    NOTE: MuJoCo convention - positive ctrl produces rotation about the joint axis.
    The regressor computes torque in the body frame. To match MuJoCo's ctrl,
    we need: tau = axis @ Y_angular @ theta (no negation needed).
    
    Args:
        regressor_6x10: The full 6x10 regressor matrix
        joint_axis: 3-element array, the joint rotation axis in the body frame
                    (e.g., [0, -1, 0] for rotation about -Y)
    
    Returns:
        1x10 regressor row for the joint torque equation
    """
    joint_axis = np.array(joint_axis).flatten()
    # Extract angular part (first 3 rows) and project onto joint axis
    Y_angular = regressor_6x10[0:3, :]  # 3x10
    Y_joint = joint_axis @ Y_angular     # 1x10
    return Y_joint


# Create the regressor function once at module load
_regressor_func = None


def get_regressor_func():
    """Get the cached regressor function (creates it on first call)."""
    global _regressor_func
    if _regressor_func is None:
        logger.info("Generating symbolic regressor (one-time cost)...")
        _regressor_func = derive_regressor_from_spatial_eqn()
        logger.info("Symbolic regressor generation complete.")
    return _regressor_func


def compute_joint_regressor_row(wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz, joint_axis):
    """
    Compute the 1x10 regressor row for a joint given kinematic data.
    
    Args:
        wx, wy, wz: Angular velocity in body frame (rad/s)
        vx, vy, vz: Linear velocity in body frame (m/s)
        dwx, dwy, dwz: Angular acceleration in body frame (rad/s^2)
        dvx, dvy, dvz: Linear acceleration in body frame (m/s^2)
        joint_axis: 3-element array, joint rotation axis in body frame
    
    Returns:
        1x10 numpy array - the regressor row for tau_joint = Y_row @ theta
    """
    regressor_func = get_regressor_func()
    Y_full = regressor_func(wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz)
    Y_row = get_joint_regressor_row(Y_full, joint_axis)
    return Y_row


def compute_joint_regressor_row_with_friction(wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz, 
                                               joint_axis, joint_velocity):
    """
    Compute the 1x12 regressor row including Coulomb + viscous friction.
    
    The model is:
        τ = Y_inertial @ θ_inertial + Fc * sign(q̇) + Fv * q̇
    
    Extended parameter vector:
        θ = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Fc, Fv]
    
    Args:
        wx, wy, wz: Angular velocity in body frame (rad/s)
        vx, vy, vz: Linear velocity in body frame (m/s)
        dwx, dwy, dwz: Angular acceleration in body frame (rad/s^2)
        dvx, dvy, dvz: Linear acceleration in body frame (m/s^2)
        joint_axis: 3-element array, joint rotation axis in body frame
        joint_velocity: Joint velocity q̇ (rad/s) - used for friction terms
    
    Returns:
        1x12 numpy array - the regressor row for tau_joint = Y_row @ theta_extended
        
    Note:
        θ_extended = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Fc, Fv]
        where Fc is Coulomb friction coefficient and Fv is viscous friction coefficient
    """
    # Get the 10-parameter inertial regressor
    Y_inertial = compute_joint_regressor_row(wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz, joint_axis)
    
    # Friction terms: τ_friction = Fc * sign(q̇) + Fv * q̇
    sign_qd = np.sign(joint_velocity)
    
    # Extended regressor: [Y_inertial, sign(q̇), q̇]
    Y_extended = np.concatenate([Y_inertial, [sign_qd, joint_velocity]])
    
    return Y_extended


def compute_joint_regressor_row_with_friction_asymmetric(wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz, 
                                                          joint_axis, joint_velocity):
    """
    Compute the 1x14 regressor row with asymmetric Coulomb + viscous friction.
    
    Some joints have different friction in positive vs negative direction.
    
    The model is:
        τ = Y_inertial @ θ_inertial + Fc+ * H(q̇) + Fc- * H(-q̇) + Fv+ * max(q̇,0) + Fv- * min(q̇,0)
    
    where H(x) is the Heaviside step function (smooth version).
    
    Extended parameter vector:
        θ = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Fc+, Fc-, Fv+, Fv-]
    
    Args:
        wx, wy, wz: Angular velocity in body frame (rad/s)
        vx, vy, vz: Linear velocity in body frame (m/s)
        dwx, dwy, dwz: Angular acceleration in body frame (rad/s^2)
        dvx, dvy, dvz: Linear acceleration in body frame (m/s^2)
        joint_axis: 3-element array, joint rotation axis in body frame
        joint_velocity: Joint velocity q̇ (rad/s)
    
    Returns:
        1x14 numpy array - the regressor row
    """
    Y_inertial = compute_joint_regressor_row(wx, wy, wz, vx, vy, vz, dwx, dwy, dwz, dvx, dvy, dvz, joint_axis)
    
    epsilon = 0.01  # rad/s
    
    # Smooth Heaviside: H(x) ≈ 0.5 * (1 + tanh(x/epsilon))
    H_pos = 0.5 * (1 + np.tanh(joint_velocity / epsilon))  # ~1 when q̇ > 0
    H_neg = 0.5 * (1 + np.tanh(-joint_velocity / epsilon))  # ~1 when q̇ < 0
    
    # Smooth positive/negative velocity
    qd_pos = joint_velocity * H_pos  # ≈ q̇ when q̇ > 0, ≈ 0 otherwise
    qd_neg = joint_velocity * H_neg  # ≈ q̇ when q̇ < 0, ≈ 0 otherwise
    
    # Extended regressor: [Y_inertial, H(q̇), H(-q̇), q̇+, q̇-]
    Y_extended = np.concatenate([Y_inertial, [H_pos, H_neg, qd_pos, qd_neg]])
    
    return Y_extended


if __name__ == "__main__":
    # Test the regressor
    print("Testing regressor generation...")
    
    # Get the function
    regressor_func = get_regressor_func()
    
    # Test with some values
    Y = regressor_func(
        wx=0.1, wy=0.2, wz=0.3,
        vx=0.0, vy=0.0, vz=0.0,
        dwx=0.01, dwy=0.02, dwz=0.03,
        dvx=0.0, dvy=0.0, dvz=9.81
    )
    print(f"Full regressor shape: {Y.shape}")
    print(f"Full regressor:\n{Y}")
    
    # Test joint row extraction for joint6 (axis = [0, -1, 0])
    joint6_axis = [0, -1, 0]
    Y_row = get_joint_regressor_row(Y, joint6_axis)
    print(f"\nJoint6 regressor row (axis={joint6_axis}):")
    print(f"Shape: {Y_row.shape}")
    print(f"Values: {Y_row}")
