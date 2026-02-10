"""Tests for enhanced CLI functionality with project integration."""

import tempfile
import pytest
from pathlib import Path
from typer.testing import CliRunner

from sidecar_os.cli.sidecar import app


class TestEnhancedCLI:
    """Test enhanced CLI commands with project integration."""

    def test_smart_add_basic_functionality(self):
        """Test basic smart add command functionality."""
        runner = CliRunner()

        result = runner.invoke(app, [
            "add", "LPD: need to run experiments"
        ])

        assert result.exit_code == 0
        assert "Added to inbox" in result.stdout

    def test_add_without_smart_interpretation(self):
        """Test add command with smart interpretation disabled."""
        runner = CliRunner()

        result = runner.invoke(app, [
            "add", "LPD: need to run experiments", "--no-smart"
        ])

        assert result.exit_code == 0
        assert "Added to inbox" in result.stdout

    def test_status_command_basic(self):
        """Test basic status command functionality."""
        runner = CliRunner()

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Sidecar OS Status" in result.stdout

    def test_focus_command_nonexistent_project(self):
        """Test focus command with nonexistent project."""
        runner = CliRunner()

        result = runner.invoke(app, ["focus", "nonexistent"])

        assert result.exit_code == 0
        assert "Project not found" in result.stdout

    def test_triage_command_basic(self):
        """Test basic triage command functionality."""
        runner = CliRunner()

        result = runner.invoke(app, ["triage"])

        assert result.exit_code == 0
        assert "Triage Mode" in result.stdout

    def test_project_add_command_basic(self):
        """Test basic project addition."""
        runner = CliRunner()

        result = runner.invoke(app, [
            "project-add", "My Project", "--alias", "mp"
        ])

        assert result.exit_code == 0
        assert "Created project" in result.stdout

    def test_project_list_command_basic(self):
        """Test basic project listing."""
        runner = CliRunner()

        result = runner.invoke(app, ["project-list"])

        assert result.exit_code == 0
        # Should show either projects or "No projects found"

    def test_help_commands_work(self):
        """Test that help is available for all new commands."""
        runner = CliRunner()

        commands = ["focus", "triage", "project-add", "project-list"]

        for command in commands:
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.stdout

    def test_add_command_smart_flag_variations(self):
        """Test add command with different smart flag variations."""
        runner = CliRunner()

        # Test with --smart (default)
        result = runner.invoke(app, ["add", "test item", "--smart"])
        assert result.exit_code == 0

        # Test with --no-smart
        result = runner.invoke(app, ["add", "test item", "--no-smart"])
        assert result.exit_code == 0

        # Test with -s shorthand
        result = runner.invoke(app, ["add", "test item", "-s"])
        assert result.exit_code == 0

        # Test with -n shorthand
        result = runner.invoke(app, ["add", "test item", "-n"])
        assert result.exit_code == 0