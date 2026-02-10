"""Tests for the main CLI application."""

import pytest
from typer.testing import CliRunner

from sidecar_os.cli.sidecar import app

runner = CliRunner()


def test_cli_help():
    """Test that CLI help shows proper information."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Sidecar OS - event-sourced productivity assistant" in result.stdout
    assert "hello" in result.stdout
    assert "add" in result.stdout
    assert "status" in result.stdout


def test_hello_command():
    """Test hello command executes successfully."""
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Sidecar OS is alive." in result.stdout


def test_add_command():
    """Test add command executes successfully."""
    result = runner.invoke(app, ["add", "test task"])
    assert result.exit_code == 0
    assert "Added to inbox: test task" in result.stdout
    assert "Event ID:" in result.stdout


def test_status_command():
    """Test status command executes successfully."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Sidecar OS Status" in result.stdout
    # Note: Status output varies based on whether any events exist


def test_add_command_with_spaces():
    """Test add command handles text with spaces correctly."""
    result = runner.invoke(app, ["add", "task with multiple words"])
    assert result.exit_code == 0
    assert "Added to inbox: task with multiple words" in result.stdout
    assert "Event ID:" in result.stdout