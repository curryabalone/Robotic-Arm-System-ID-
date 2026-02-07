# robot-sysid

A generic robotic arm system identification tool that identifies the inertial parameters and friction of a robot's terminal (end-effector) link from MuJoCo simulation data.

## Overview

This tool automates the process of identifying the 10 standard rigid-body inertial parameters (mass, center-of-mass moments, and inertia tensor) plus optional Coulomb and viscous friction coefficients for the last link of a robotic arm. It works with any robot described by a MuJoCo XML model.

**Common use case:** You've attached a payload or tool to your robot's end-effector and need to identify the composite inertial properties of that terminal link plus payload for accurate dynamic modeling and control.

### System Identification Approach

The tool implements a physics-based system identification pipeline:

1. **Parse** — Load a MuJoCo XML model and extract the kinematic chain, identifying the terminal (end-effector) joint automatically
2. **Generate** — Create multi-frequency sinusoidal excitation trajectories that richly excite the terminal link's dynamics while respecting joint limits
3. **Simulate** — Run MuJoCo simulation along the trajectory, collecting spatial velocity, acceleration, and torque data at each timestep
4. **Solve** — Compute the regressor matrix Y from spatial dynamics such that $\tau = Y \theta$, then solve for parameters via least-squares: $\theta = Y^{\dagger} \tau$
5. **Export** — Output identified parameters as JSON, generate an updated MuJoCo XML with corrected inertias, and optionally export a trajectory CSV for hardware playback on Damiao motors

The spatial dynamics regressor Y is derived symbolically via SymPy, ensuring analytical correctness. The regressor relates joint torques to the 10 inertial parameters (mass, first moment of mass, inertia tensor) plus 2 friction coefficients (Coulomb and viscous).

### Key Features

- **Robot-agnostic** — Works with any MuJoCo XML model, not limited to specific robot types
- **Automatic terminal joint detection** — Identifies the end-effector joint automatically from the kinematic tree
- **Analytical derivatives** — Trajectory velocities and accelerations are computed exactly, not via numerical differentiation
- **Symbolic regressor derivation** — Uses SymPy to derive the spatial dynamics regressor, ensuring mathematical correctness
- **Friction identification** — Optionally identifies Coulomb and viscous friction coefficients for the terminal joint
- **Hardware export** — Generates trajectory files for Damiao motors with automatic limit clamping
- **Quality metrics** — Reports RMSE, condition number, and matrix rank to assess identification quality

## Installation

### Prerequisites

- Python 3.9 or later
- MuJoCo 3.0 or later

### Option 1: Install from Source (pip)

```bash
# Clone the repository
git clone https://github.com/curryabalone/Robotic-Arm-System-ID-.git
cd Robotic-Arm-System-ID-

# Install in editable mode
pip install -e .
```

This installs the `robot-sysid` command-line tool and all required dependencies (numpy, mujoco, sympy, scipy).

### Option 2: Install with Conda Environment

```bash
# Clone the repository
git clone https://github.com/curryabalone/Robotic-Arm-System-ID-.git
cd Robotic-Arm-System-ID-

# Create and activate conda environment
conda env create -f environment.yml
conda activate robot-sysid

# Install the package
pip install -e .
```

### Install with Development Dependencies

For running tests and contributing to development:

```bash
pip install -e ".[dev]"
```

This adds pytest and hypothesis for unit and property-based testing.

## Quick Start

Run system identification on the included Kinova arm example:

```bash
robot-sysid examples/kinova/model/kinova_fullinertia_guess.xml
```

**What happens:**
1. Loads the Kinova 7-DOF arm model with guessed inertial parameters
2. Generates a 15-second multi-frequency sinusoidal excitation trajectory for the terminal joint (joint6)
3. Runs MuJoCo simulation to collect regressor matrix Y and torque vector τ
4. Solves for the 10 inertial parameters + 2 friction coefficients via least-squares
5. Saves results to `./sysid_output/`:
   - `params.json` — Identified parameters in JSON format
   - `updated_model.xml` — MuJoCo XML with corrected inertial parameters

**Expected output:**
```
Loading robot model from examples/kinova/model/kinova_fullinertia_guess.xml...
Identified terminal joint: joint6
Generating excitation trajectory (15.0s, 1000 Hz)...
Running MuJoCo simulation...
Collecting regressor data: 100%
Solving for parameters...
Identified parameters:
  Mass: 0.789 kg
  CoM moment: [-0.000, 0.009, 0.120] kg·m
  Inertia: [0.024, 0.000, 0.000, 0.024, -0.002, 0.001] kg·m²
  Coulomb friction: 0.40 N·m
  Viscous friction: 0.10 N·m·s/rad
  RMSE: 0.0001 N·m
  Condition number: 1234.5
  Base parameters: 7
Results saved to ./sysid_output/
```

## Usage

### Basic Command

```bash
robot-sysid <path-to-mujoco-xml> [options]
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mesh-dir MESH_DIR` | auto | Path to mesh directory (inferred from XML `meshdir` attribute if not specified) |
| `--duration DURATION` | 15 | Trajectory duration in seconds (longer = better conditioning) |
| `--sample-rate RATE` | 1000 | Sample rate in Hz (higher = more data points) |
| `--no-friction` | — | Disable friction identification (identify only 10 inertial params) |
| `--output-dir DIR` | `./sysid_output` | Output directory for results |
| `--export-damiao` | — | Export trajectory CSV for Damiao motor playback |
| `--motor-type TYPE` | — | Damiao motor type per joint (e.g., DM4340). Repeat for each joint when using `--export-damiao` |
| `--seed SEED` | — | Random seed for reproducible trajectory generation |

### Usage Examples

#### Identify Only Inertial Parameters (No Friction)

```bash
robot-sysid examples/kinova/model/kinova_fullinertia_guess.xml --no-friction
```

Solves for 10 parameters: mass, first moment of mass (3), inertia tensor (6).

#### Custom Trajectory Duration and Sample Rate

```bash
robot-sysid examples/kinova/model/kinova_fullinertia_guess.xml --duration 30 --sample-rate 500
```

Uses a 30-second trajectory sampled at 500 Hz. Longer trajectories improve regressor conditioning but take more time to simulate.

#### Specify Output Directory

```bash
robot-sysid examples/kinova/model/kinova_fullinertia_guess.xml --output-dir results/kinova_sysid
```

Saves results to `results/kinova_sysid/` instead of the default `./sysid_output/`.

#### Export Trajectory for Damiao Motors

```bash
robot-sysid examples/kinova/model/kinova_fullinertia_guess.xml \
  --export-damiao \
  --motor-type DM4340 \
  --motor-type DM4340 \
  --motor-type DM4340 \
  --motor-type DM4340 \
  --motor-type DM4340 \
  --motor-type DM4340 \
  --motor-type DM4340
```

Generates `damiao_trajectory.csv` with timestamped position, velocity, and feedforward torque commands for each joint, clamped to the specified motor limits. Useful for running the same trajectory on real hardware.

**Supported Damiao motor types:** DM4310, DM4340, DM4340_48V, DM6006, DM8006, DM10010L, DM8009

#### Reproducible Results

```bash
robot-sysid examples/kinova/model/kinova_fullinertia_guess.xml --seed 42
```

Uses a fixed random seed for deterministic trajectory generation, ensuring reproducible results across runs.

## Output Formats

### `params.json` — Identified Parameters

JSON file containing the identified parameters and quality metrics:

```json
{
  "joint_name": "joint6",
  "mass": 0.789160,
  "center_of_mass_moment": [-0.000047, 0.009093, 0.120437],
  "inertia_tensor": {
    "Ixx": 0.024382,
    "Ixy": 0.000001,
    "Ixz": 0.000009,
    "Iyy": 0.024312,
    "Iyz": -0.001638,
    "Izz": 0.000551
  },
  "coulomb_friction": 0.4,
  "viscous_friction": 0.1,
  "rmse": 0.000123,
  "condition_number": 1234.56,
  "n_base_params": 7
}
```

**Parameter definitions:**
- **mass** — Total mass of the terminal link (kg)
- **center_of_mass_moment** — First moment of mass [hx, hy, hz] = m × [cx, cy, cz] (kg·m)
- **inertia_tensor** — 6 independent components [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] about the body frame origin (kg·m²)
- **coulomb_friction** — Constant friction torque (N·m)
- **viscous_friction** — Velocity-dependent friction coefficient (N·m·s/rad)

**Quality metrics:**
- **rmse** — Root mean square error between measured and predicted torques (N·m). Lower is better. Typical values: < 0.01 N·m for simulation data.
- **condition_number** — Measure of regressor matrix conditioning. Lower is better. Values > 1000 may indicate poor excitation.
- **n_base_params** — Number of identifiable parameter combinations (matrix rank). Should be close to 10 (or 12 with friction) for well-conditioned data.

### `updated_model.xml` — Updated MuJoCo Model

A copy of the input MuJoCo XML with the terminal body's inertial parameters replaced by the identified values. All other elements (meshes, actuators, sensors, visual elements) are preserved.

The updated XML includes a comment noting which body was identified and the identification RMSE:

```xml
<!-- System ID: body 'link6' identified with RMSE 0.000123 Nm -->
<body name="link6">
  <inertial pos="0.0 0.0115 0.1526" mass="0.789160"
            fullinertia="0.024382 0.024312 0.000551 0.000001 0.000009 -0.001638"/>
  ...
</body>
```

### `damiao_trajectory.csv` — Motor Trajectory (Optional)

Generated when `--export-damiao` is specified. Contains timestamped rows with position, velocity, and feedforward torque for each joint:

```csv
# motor_type=DM4340,can_id=1,mode=MIT
# motor_type=DM4340,can_id=2,mode=MIT
# motor_type=DM4340,can_id=3,mode=MIT
# ...
time,j0_pos,j0_vel,j0_tau,j1_pos,j1_vel,j1_tau,...
0.000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,...
0.001,0.0012,0.3100,0.0500,0.0008,0.2200,0.0300,...
0.002,0.0024,0.3150,0.0510,0.0016,0.2250,0.0305,...
```

Values are automatically clamped to the specified Damiao motor limits. The tool reports the percentage of clamped points per joint.

## Understanding the System Identification Approach

### Spatial Dynamics Regressor

The tool uses the spatial dynamics formulation to derive a linear relationship between joint torques and inertial parameters:

$$\tau = Y \theta$$

Where:
- $\tau$ — Joint torque (scalar for a single joint)
- $Y$ — Regressor matrix (n_samples × 10 without friction, or n_samples × 12 with friction)
- $\theta$ — Parameter vector: $[m, h_x, h_y, h_z, I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}]$ for inertial-only, or $[m, h_x, h_y, h_z, I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}, f_c, f_v]$ with friction

The regressor Y is derived from the spatial dynamics equation:

$$\mathbf{F}_a = \mathbf{G}_a \dot{\mathbf{V}}_a - [\text{ad}_{\mathbf{V}_a}]^T \mathbf{G}_a \mathbf{V}_a$$

Where:
- $\mathbf{F}_a$ — 6D wrench (force + torque): $\mathbf{F}_a = \begin{bmatrix} \boldsymbol{\tau} \\ \mathbf{f} \end{bmatrix} \in \mathbb{R}^6$
- $\mathbf{G}_a$ — Spatial inertia matrix (function of $\theta$)
- $\mathbf{V}_a$ — Spatial velocity (angular + linear): $\mathbf{V}_a = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix} \in \mathbb{R}^6$
- $\dot{\mathbf{V}}_a$ — Spatial acceleration (angular + linear): $\dot{\mathbf{V}}_a = \begin{bmatrix} \dot{\boldsymbol{\omega}} \\ \dot{\mathbf{v}} \end{bmatrix} \in \mathbb{R}^6$
- $[\text{ad}_{\mathbf{V}_a}]$ — Adjoint operator (captures gyroscopic effects)

The 6D wrench is projected onto the joint axis to obtain the scalar joint torque. The regressor is derived symbolically using SymPy and cached for efficient evaluation.

### Full Regressor Matrix Derivation

The spatial inertia matrix $\mathbf{G}_a \in \mathbb{R}^{6 \times 6}$ is defined as:

$$\mathbf{G}_a = \begin{bmatrix} \bar{\mathbf{I}} & [\mathbf{h}]_\times \\ -[\mathbf{h}]_\times & m\mathbf{I}_3 \end{bmatrix}$$

where:
- $\bar{\mathbf{I}} = \begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\ I_{xy} & I_{yy} & I_{yz} \\ I_{xz} & I_{yz} & I_{zz} \end{bmatrix}$ is the inertia tensor
- $\mathbf{h} = \begin{bmatrix} h_x \\ h_y \\ h_z \end{bmatrix} = m \mathbf{c}$ is the first moment of mass (mass times center of mass)
- $[\mathbf{h}]_\times = \begin{bmatrix} 0 & -h_z & h_y \\ h_z & 0 & -h_x \\ -h_y & h_x & 0 \end{bmatrix}$ is the skew-symmetric matrix

The adjoint transpose $[\text{ad}_{\mathbf{V}_a}]^T \in \mathbb{R}^{6 \times 6}$ is:

$$[\text{ad}_{\mathbf{V}_a}]^T = \begin{bmatrix} -[\boldsymbol{\omega}]_\times & -[\mathbf{v}]_\times \\ \mathbf{0}_{3 \times 3} & -[\boldsymbol{\omega}]_\times \end{bmatrix}$$

where $[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$ and $[\mathbf{v}]_\times$ is defined similarly.

The full $6 \times 10$ regressor matrix $\mathbf{Y}$ is obtained by computing the Jacobian $\frac{\partial \mathbf{F}_a}{\partial \theta}$, where each element $Y_{ij} = \frac{\partial F_{a,i}}{\partial \theta_j}$.

The complete regressor matrix with all 60 terms is:

$$\mathbf{Y} = \begin{bmatrix}
0 & 0 & \dot{v}_z + \omega_x v_y - \omega_y v_x & -\dot{v}_y + \omega_x v_z - \omega_z v_x & \dot{\omega}_x & \dot{\omega}_y - \omega_x \omega_z & \dot{\omega}_z + \omega_x \omega_y & -\omega_y \omega_z & \omega_y^2 - \omega_z^2 & \omega_y \omega_z \\
0 & -\dot{v}_z - \omega_x v_y + \omega_y v_x & 0 & \dot{v}_x + \omega_y v_z - \omega_z v_y & \omega_x \omega_z & \dot{\omega}_x + \omega_y \omega_z & -\omega_x^2 + \omega_z^2 & \dot{\omega}_y & \dot{\omega}_z - \omega_x \omega_y & -\omega_x \omega_z \\
0 & \dot{v}_y - \omega_x v_z + \omega_z v_x & -\dot{v}_x - \omega_y v_z + \omega_z v_y & 0 & -\omega_x \omega_y & \omega_x^2 - \omega_y^2 & \dot{\omega}_x - \omega_y \omega_z & \omega_x \omega_y & \dot{\omega}_y + \omega_x \omega_z & \dot{\omega}_z \\
\dot{v}_x + \omega_y v_z - \omega_z v_y & -\omega_y^2 - \omega_z^2 & -\dot{\omega}_z + \omega_x \omega_y & \dot{\omega}_y + \omega_x \omega_z & 0 & 0 & 0 & 0 & 0 & 0 \\
\dot{v}_y - \omega_x v_z + \omega_z v_x & \dot{\omega}_z + \omega_x \omega_y & -\omega_x^2 - \omega_z^2 & -\dot{\omega}_x + \omega_y \omega_z & 0 & 0 & 0 & 0 & 0 & 0 \\
\dot{v}_z + \omega_x v_y - \omega_y v_x & -\dot{\omega}_y + \omega_x \omega_z & \dot{\omega}_x + \omega_y \omega_z & -\omega_x^2 - \omega_y^2 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}$$

where:
- Row 1: Torque component about x-axis ($n_x$)
- Row 2: Torque component about y-axis ($n_y$)  
- Row 3: Torque component about z-axis ($n_z$)
- Row 4: Force component along x-axis ($f_x$)
- Row 5: Force component along y-axis ($f_y$)
- Row 6: Force component along z-axis ($f_z$)

And columns correspond to parameters:
- Column 1: Mass $m$
- Column 2: First moment $h_x$
- Column 3: First moment $h_y$
- Column 4: First moment $h_z$
- Column 5: Inertia $I_{xx}$
- Column 6: Inertia $I_{xy}$
- Column 7: Inertia $I_{xz}$
- Column 8: Inertia $I_{yy}$
- Column 9: Inertia $I_{yz}$
- Column 10: Inertia $I_{zz}$

Each row corresponds to one component of the 6D wrench $\mathbf{F}_a = [n_x, n_y, n_z, f_x, f_y, f_z]^T$, and each column corresponds to one inertial parameter in $\theta = [m, h_x, h_y, h_z, I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}]^T$.

For a revolute joint with axis $\mathbf{a} = [a_x, a_y, a_z]^T$, the scalar joint torque is obtained by projecting the angular torque components onto the joint axis:

$$\tau = \mathbf{a}^T \begin{bmatrix} n_x \\ n_y \\ n_z \end{bmatrix} = \mathbf{a}^T \begin{bmatrix} \mathbf{Y}_{1,1:10} \\ \mathbf{Y}_{2,1:10} \\ \mathbf{Y}_{3,1:10} \end{bmatrix} \theta$$

where $\mathbf{Y}_{i,1:10}$ denotes row $i$ of the regressor matrix (the first three rows contain the angular torque components).

This gives a $1 \times 10$ regressor row for each timestep. When friction is included, two additional columns are appended:

$$\mathbf{Y}_{\text{friction}} = \begin{bmatrix} \mathbf{Y}_{\text{inertial}} & \text{sign}(\dot{q}) & \dot{q} \end{bmatrix}$$

giving the extended parameter vector $\theta_{\text{ext}} = [m, h_x, h_y, h_z, I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}, f_c, f_v]^T$, where $f_c$ is Coulomb friction and $f_v$ is viscous friction.

The complete friction model is:

$$\tau_{\text{friction}} = f_c \cdot \text{sign}(\dot{q}) + f_v \cdot \dot{q}$$

And the total joint torque becomes:

$$\tau_{\text{total}} = \mathbf{a}^T \begin{bmatrix} \mathbf{Y}_{1,1:10} \\ \mathbf{Y}_{2,1:10} \\ \mathbf{Y}_{3,1:10} \end{bmatrix} \theta_{\text{inertial}} + f_c \cdot \text{sign}(\dot{q}) + f_v \cdot \dot{q}$$

where the first term represents the inertial torque contribution and the latter terms represent Coulomb and viscous friction.

### Excitation Trajectory Design

The tool generates multi-frequency sinusoidal trajectories to ensure rich excitation:

- **Terminal joint** — Sum of 3–5 sinusoids with frequencies 0.1–0.6 Hz and amplitudes scaled to stay within joint limits
- **Non-terminal joints** — Low-amplitude multi-frequency sinusoids to improve regressor conditioning
- **Incommensurate frequencies** — Frequencies are chosen to avoid periodicity and ensure diverse kinematic states

Velocities and accelerations are computed analytically (exact derivatives of position sinusoids), avoiding numerical differentiation errors.

### Least-Squares Solution

The parameter vector θ is recovered via least-squares:

$$\theta = Y^{\dagger} \tau$$

Where $Y^{\dagger}$ is the Moore-Penrose pseudoinverse. The tool uses `np.linalg.lstsq` with `rcond=1e-10` for robust least-squares, which handles rank-deficient regressor matrices gracefully.

### Base Parameters

Due to the structure of the dynamics equations, not all 10 inertial parameters are independently identifiable from single-joint torque measurements. The identifiable combinations are called **base parameters**. The tool reports the number of base parameters (matrix rank), which should be close to 10 for well-conditioned data.

## Project Structure

```
robot-sysid/
├── robot_sysid/          # Main package
│   ├── __init__.py       # Package initialization
│   ├── cli.py            # CLI entry point (argparse, pipeline orchestration)
│   ├── parser.py         # MuJoCo XML model loading and validation
│   ├── trajectory.py     # Excitation trajectory generation
│   ├── regressor.py      # Spatial dynamics regressor computation (SymPy)
│   ├── simulator.py      # MuJoCo simulation and data collection
│   ├── solver.py         # Least-squares parameter identification
│   └── export.py         # JSON, XML, and CSV exporters
├── examples/
│   └── kinova/           # Kinova arm reference example
│       ├── model/        # MuJoCo XML models and STL meshes
│       ├── validate_*.py # Validation scripts
│       └── README.md     # Kinova example documentation
├── tests/                # Unit and property-based tests
│   ├── test_parser.py
│   ├── test_trajectory.py
│   ├── test_regressor.py
│   ├── test_simulator.py
│   ├── test_solver.py
│   └── test_export.py
├── pyproject.toml        # Package metadata and dependencies
└── README.md             # This file
```

## Troubleshooting

### High RMSE (> 0.1 N·m)

**Possible causes:**
- Insufficient excitation — trajectory may not be rich enough
- Poor initial inertial guesses in the XML
- Incorrect mesh references or joint definitions

**Solutions:**
- Increase trajectory duration: `--duration 30`
- Try a different random seed: `--seed 123`
- Verify that the model's joint limits are reasonable
- Check that mesh files are correctly referenced

### Low Matrix Rank (< 7 base parameters)

**Possible causes:**
- Insufficient excitation — regressor matrix is poorly conditioned
- Trajectory frequencies are too similar (near-periodic motion)

**Solutions:**
- Increase trajectory duration: `--duration 20`
- Try a different random seed to get different frequencies: `--seed 456`
- Check that non-terminal joints are moving (they should have low-amplitude excitation)

### Simulation Divergence (NaN values)

**Possible causes:**
- Unreasonable initial inertial guesses (e.g., negative mass, non-positive-definite inertia)
- Joint limits are too restrictive, causing large control forces
- Trajectory amplitude is too large

**Solutions:**
- Check that the initial model has physically plausible inertial parameters
- Verify that joint limits in the XML are not overly restrictive
- Reduce trajectory amplitude by tightening joint limits in the XML

### Clamping Warnings (Damiao Export)

If many trajectory points are clamped when exporting for Damiao motors:
- The trajectory may exceed motor limits
- Consider using a motor with higher limits (e.g., DM6006 instead of DM4340)
- Reduce trajectory duration or amplitude by adjusting joint limits in the XML

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_solver.py

# Run with verbose output
pytest tests/ -v

# Run property-based tests with more examples
pytest tests/ --hypothesis-seed=42
```

### Test Organization

- **Unit tests** — Test specific examples, edge cases, and error conditions using pytest
- **Property-based tests** — Test universal correctness properties using hypothesis (100+ examples per property)
- **Integration tests** — End-to-end tests using the Kinova example model

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## Examples

See the `examples/kinova/` directory for a complete reference example demonstrating how to use the tool with a Kinova 7-DOF robotic arm. The example includes:

- MuJoCo XML models with ground truth and guessed inertial parameters
- Validation scripts demonstrating the full pipeline
- Detailed README with usage examples and output interpretation

For more details, see [`examples/kinova/README.md`](examples/kinova/README.md).

## License

MIT

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{robot_sysid,
  title = {robot-sysid: A Generic Robot System Identification Tool},
  author = {Cai, Yuze},
  year = {2026},
  url = {https://github.com/curryabalone/Robotic-Arm-System-ID-}
}
```
