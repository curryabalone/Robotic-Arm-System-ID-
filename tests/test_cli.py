"""Tests for CLI argument parsing and validation."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI with the given arguments and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "robot_sysid.cli", *args],
        capture_output=True,
        text=True,
    )


class TestCLIVersion:
    """Tests for the --version flag."""

    def test_version_flag(self):
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "robot-sysid" in result.stdout
        assert "0.1.0" in result.stdout


class TestCLIValidation:
    """Tests for input validation in the CLI."""

    def test_rejects_non_xml_file(self):
        result = _run_cli("model.txt")
        assert result.returncode != 0
        assert "Expected an XML model file" in result.stderr

    def test_accepts_xml_extension(self):
        # Will fail later (file not found), but should pass extension check
        result = _run_cli("model.xml")
        assert "Expected an XML model file" not in result.stderr

    def test_accepts_mjcf_extension(self):
        result = _run_cli("model.mjcf")
        assert "Expected an XML model file" not in result.stderr

    def test_rejects_zero_duration(self):
        result = _run_cli("model.xml", "--duration", "0")
        assert result.returncode != 0
        assert "--duration must be positive" in result.stderr

    def test_rejects_negative_duration(self):
        result = _run_cli("model.xml", "--duration", "-5")
        assert result.returncode != 0
        assert "--duration must be positive" in result.stderr

    def test_rejects_zero_sample_rate(self):
        result = _run_cli("model.xml", "--sample-rate", "0")
        assert result.returncode != 0
        assert "--sample-rate must be positive" in result.stderr

    def test_rejects_negative_sample_rate(self):
        result = _run_cli("model.xml", "--sample-rate", "-100")
        assert result.returncode != 0
        assert "--sample-rate must be positive" in result.stderr

    def test_damiao_requires_motor_type(self):
        result = _run_cli("model.xml", "--export-damiao")
        assert result.returncode != 0
        assert "--export-damiao requires at least one --motor-type" in result.stderr
