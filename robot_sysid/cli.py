"""Command-line interface for robot system identification.

Provides a CLI entry point that wires together the full system identification
pipeline: load model → generate trajectory → simulate → identify → export.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from robot_sysid import __version__
from robot_sysid.export import (
    export_damiao_trajectory,
    export_params_json,
    export_updated_xml,
)
from robot_sysid.parser import load_robot_model
from robot_sysid.simulator import collect_sysid_data
from robot_sysid.solver import identify_parameters
from robot_sysid.trajectory import generate_excitation_trajectory


def main() -> None:
    """CLI entry point using argparse.

    Parses command-line arguments and runs the full system identification
    pipeline:
    1. Load robot model from MuJoCo XML
    2. Generate excitation trajectory
    3. Run MuJoCo simulation to collect regressor data
    4. Identify inertial parameters and friction
    5. Export results (JSON, updated XML, optional Damiao CSV)

    The CLI displays progress messages during each step and saves all outputs
    to the specified output directory.
    """
    parser = argparse.ArgumentParser(
        prog="robot-sysid",
        description=(
            "Robot system identification tool — identifies inertial parameters "
            "and friction of a robot's terminal link from MuJoCo simulation data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  robot-sysid model.xml

  # Specify mesh directory and output location
  robot-sysid model.xml --mesh-dir meshes/ --output-dir results/

  # Disable friction identification
  robot-sysid model.xml --no-friction

  # Export Damiao trajectory for two joints
  robot-sysid model.xml --export-damiao --motor-type DM4340 --motor-type DM4340

  # Custom trajectory duration and sample rate
  robot-sysid model.xml --duration 20 --sample-rate 500
        """,
    )

    # Positional arguments
    parser.add_argument(
        "xml_path",
        type=str,
        help="Path to MuJoCo XML model file",
    )

    # Optional arguments
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help=(
            "Path to mesh directory (default: inferred from XML meshdir "
            "compiler attribute, or not validated if not specified)"
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Trajectory duration in seconds (default: 15)",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1000,
        help="Sample rate in Hz (default: 1000)",
    )

    parser.add_argument(
        "--no-friction",
        action="store_true",
        help="Disable friction identification (default: friction is identified)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sysid_output",
        help="Output directory for results (default: ./sysid_output)",
    )

    parser.add_argument(
        "--export-damiao",
        action="store_true",
        help="Enable Damiao trajectory export (requires --motor-type)",
    )

    parser.add_argument(
        "--motor-type",
        type=str,
        action="append",
        dest="motor_types",
        help=(
            "Damiao motor type per joint (e.g., DM4340). Can be repeated for "
            "multiple joints. Required if --export-damiao is specified."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for trajectory generation (default: None)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug-level) logging output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Validate arguments
    if args.export_damiao and not args.motor_types:
        parser.error("--export-damiao requires at least one --motor-type")

    xml_path = Path(args.xml_path)
    if xml_path.suffix.lower() not in (".xml", ".mjcf"):
        parser.error(
            f"Expected an XML model file (.xml or .mjcf), got '{xml_path.suffix}'"
        )

    if args.duration <= 0:
        parser.error("--duration must be positive")

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")

    include_friction = not args.no_friction

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Robot System Identification Tool")
    print("=" * 70)
    print()

    # Step 1: Load robot model
    print(f"[1/5] Loading robot model from '{args.xml_path}'...")
    try:
        model = load_robot_model(args.xml_path, args.mesh_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ Loaded {model.n_joints} joints")
    print(f"  ✓ Terminal joint: {model.terminal_joint.name}")
    print(f"  ✓ Terminal body: {model.terminal_body.name}")
    print()

    # Step 2: Generate excitation trajectory
    print(f"[2/5] Generating excitation trajectory...")
    print(f"  Duration: {args.duration}s, Sample rate: {args.sample_rate} Hz")
    try:
        trajectory = generate_excitation_trajectory(
            model,
            duration=args.duration,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Error generating trajectory: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ Generated {len(trajectory.time)} samples")
    print()

    # Step 3: Run simulation and collect data
    print(f"[3/5] Running MuJoCo simulation...")
    print(f"  Friction identification: {'enabled' if include_friction else 'disabled'}")
    try:
        data = collect_sysid_data(
            args.xml_path,
            trajectory,
            model.terminal_joint,
            model.terminal_body.name,
            include_friction=include_friction,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ Collected {data.Y.shape[0]} samples")
    print(f"  ✓ Regressor matrix: {data.Y.shape[0]} × {data.Y.shape[1]}")
    print(f"  ✓ Condition number: {data.condition_number:.2f}")
    print(f"  ✓ Matrix rank: {data.matrix_rank}")
    print()

    # Step 4: Identify parameters
    print(f"[4/5] Identifying parameters...")
    try:
        params = identify_parameters(
            data,
            model.terminal_joint.name,
            include_friction=include_friction,
        )
    except ValueError as e:
        print(f"Error identifying parameters: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ Mass: {params.mass:.6f} kg")
    print(f"  ✓ CoM moment: [{params.com_moment[0]:.6f}, "
          f"{params.com_moment[1]:.6f}, {params.com_moment[2]:.6f}] kg·m")
    print(f"  ✓ Inertia: Ixx={params.inertia[0]:.6f}, Iyy={params.inertia[3]:.6f}, "
          f"Izz={params.inertia[5]:.6f} kg·m²")
    if params.coulomb_friction is not None:
        print(f"  ✓ Coulomb friction: {params.coulomb_friction:.6f} N·m")
    if params.viscous_friction is not None:
        print(f"  ✓ Viscous friction: {params.viscous_friction:.6f} N·m·s/rad")
    print(f"  ✓ RMSE: {params.rmse:.6f} N·m")
    print()

    # Step 5: Export results
    print(f"[5/5] Exporting results to '{output_dir}'...")

    # Export params.json
    params_json_path = output_dir / "params.json"
    try:
        export_params_json(params, str(params_json_path))
        print(f"  ✓ Saved parameters: {params_json_path}")
    except IOError as e:
        print(f"Error exporting params.json: {e}", file=sys.stderr)
        sys.exit(1)

    # Export updated_model.xml
    updated_xml_path = output_dir / "updated_model.xml"
    try:
        export_updated_xml(args.xml_path, params, str(updated_xml_path))
        print(f"  ✓ Saved updated model: {updated_xml_path}")
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error exporting updated XML: {e}", file=sys.stderr)
        sys.exit(1)

    # Export Damiao trajectory if requested
    if args.export_damiao:
        # Validate motor_types length
        if len(args.motor_types) != model.n_joints:
            print(
                f"Error: --motor-type specified {len(args.motor_types)} times, "
                f"but model has {model.n_joints} joints",
                file=sys.stderr,
            )
            sys.exit(1)

        # Generate CAN IDs (1, 2, 3, ...)
        can_ids = list(range(1, model.n_joints + 1))

        damiao_csv_path = output_dir / "damiao_trajectory.csv"
        try:
            clamping_stats = export_damiao_trajectory(
                trajectory,
                args.motor_types,
                can_ids,
                str(damiao_csv_path),
            )
            print(f"  ✓ Saved Damiao trajectory: {damiao_csv_path}")

            # Report clamping statistics
            for joint_idx in range(model.n_joints):
                pct = clamping_stats.get(f"joint{joint_idx}_clamped_pct", 0.0)
                if pct > 0.0:
                    print(f"    ⚠ Joint {joint_idx}: {pct:.1f}% of points clamped")

        except (ValueError, IOError) as e:
            print(f"Error exporting Damiao trajectory: {e}", file=sys.stderr)
            sys.exit(1)

    print()
    print("=" * 70)
    print("System identification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
