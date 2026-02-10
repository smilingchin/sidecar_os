"""Integration tests for CLI commands with event system."""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sidecar_os.cli.sidecar import app
from sidecar_os.core.sidecar_core.events import EventStore


class TestCoreCommandIntegration:
    """Test CLI commands integration with event system."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_add_command_creates_event(self, runner, temp_data_dir):
        """Test that add command creates and persists events."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            # Mock EventStore to use temp directory
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Run add command
            result = runner.invoke(app, ["add", "Test inbox item"])

            # Check command succeeded
            assert result.exit_code == 0
            assert "✓ Added to inbox: Test inbox item" in result.stdout
            assert "Event ID:" in result.stdout

            # Verify event was persisted
            events = mock_store.read_all()
            assert len(events) == 1
            assert events[0].event_type == "inbox_captured"
            assert events[0].payload["text"] == "Test inbox item"

    def test_add_command_handles_whitespace(self, runner, temp_data_dir):
        """Test that add command trims whitespace from input."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Run add command with extra whitespace
            result = runner.invoke(app, ["add", "  Whitespace test  "])

            # Check command succeeded and trimmed text
            assert result.exit_code == 0
            assert "✓ Added to inbox: Whitespace test" in result.stdout

            # Verify event contains trimmed text
            events = mock_store.read_all()
            assert events[0].payload["text"] == "Whitespace test"

    def test_status_command_empty_state(self, runner, temp_data_dir):
        """Test status command with no events."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Run status command
            result = runner.invoke(app, ["status"])

            # Check command succeeded
            assert result.exit_code == 0
            assert "📊 Sidecar OS Status" in result.stdout
            assert "System ready - no data yet" in result.stdout
            assert "0" in result.stdout  # Should show zero counts

    def test_status_command_with_inbox_items(self, runner, temp_data_dir):
        """Test status command with inbox items."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Add some events first
            runner.invoke(app, ["add", "First item"])
            runner.invoke(app, ["add", "Second item"])

            # Run status command
            result = runner.invoke(app, ["status"])

            # Check command succeeded
            assert result.exit_code == 0
            assert "📊 Sidecar OS Status" in result.stdout
            assert "📥 Inbox Items" in result.stdout
            assert "2" in result.stdout  # Should show 2 inbox items
            assert "Recent Inbox Items" in result.stdout
            assert "First item" in result.stdout
            assert "Second item" in result.stdout

    def test_full_workflow_integration(self, runner, temp_data_dir):
        """Test complete workflow from add to status."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Add multiple items
            items = ["Task 1", "Task 2", "Task 3"]
            for item in items:
                result = runner.invoke(app, ["add", item])
                assert result.exit_code == 0

            # Check status shows all items
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0

            # Verify counts
            assert "📥 Inbox Items" in result.stdout
            for item in items:
                assert item in result.stdout

            # Verify event persistence across operations
            events = mock_store.read_all()
            assert len(events) == 3
            for i, event in enumerate(events):
                assert event.payload["text"] == items[i]

    def test_event_ids_are_unique(self, runner, temp_data_dir):
        """Test that each add command creates unique event IDs."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Add multiple items
            results = []
            for i in range(3):
                result = runner.invoke(app, ["add", f"Item {i}"])
                results.append(result.stdout)
                assert result.exit_code == 0

            # Extract event IDs from outputs
            event_ids = []
            for output in results:
                lines = output.split('\n')
                for line in lines:
                    if "Event ID:" in line:
                        event_id = line.split("Event ID: ")[1].split("...")[0]
                        event_ids.append(event_id)
                        break

            # Verify all event IDs are unique
            assert len(event_ids) == 3
            assert len(set(event_ids)) == 3  # All unique

    def test_event_persistence_across_commands(self, runner, temp_data_dir):
        """Test that events persist between different command invocations."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            # Create actual event store that persists to temp directory
            def create_store():
                return EventStore(temp_data_dir)
            mock_store_class.side_effect = create_store

            # Add item in first command
            result1 = runner.invoke(app, ["add", "Persistent item"])
            assert result1.exit_code == 0

            # Check status in second command (new EventStore instance)
            result2 = runner.invoke(app, ["status"])
            assert result2.exit_code == 0
            assert "Persistent item" in result2.stdout
            assert "📥 Inbox Items" in result2.stdout

    def test_status_display_formatting(self, runner, temp_data_dir):
        """Test that status command displays formatted output correctly."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            # Add an item
            runner.invoke(app, ["add", "Format test item"])

            # Run status command
            result = runner.invoke(app, ["status"])

            # Check for expected formatting elements
            assert result.exit_code == 0
            assert "📊 Sidecar OS Status" in result.stdout
            assert "📥 Inbox Items" in result.stdout
            assert "🔄 Unprocessed" in result.stdout
            assert "📊 Total Events" in result.stdout
            assert "Recent Inbox Items" in result.stdout

            # Check that unprocessed items show the 🔄 symbol
            assert "🔄 Format test item" in result.stdout