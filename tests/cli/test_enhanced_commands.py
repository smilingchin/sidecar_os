"""Tests for enhanced CLI commands (task, list, done)."""

import tempfile
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sidecar_os.cli.sidecar import app
from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent, TaskCreatedEvent


class TestEnhancedCommands:
    """Test enhanced CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def _create_store_with_data(self, temp_data_dir):
        """Helper method to create an EventStore with sample data."""
        store = EventStore(temp_data_dir)

        # Add some inbox items
        inbox_event1 = InboxCapturedEvent(payload={"text": "First inbox item", "priority": "high"})
        inbox_event2 = InboxCapturedEvent(payload={"text": "Second inbox item", "priority": "normal"})

        store.append(inbox_event1)
        store.append(inbox_event2)

        # Add a task
        task_event = TaskCreatedEvent(payload={
            "task_id": "test_task_1",
            "title": "Test Task",
            "description": "A test task",
            "created_from_event": inbox_event1.event_id,
            "priority": "high"
        })
        store.append(task_event)

        return store, {
            "inbox1": inbox_event1,
            "inbox2": inbox_event2,
            "task1": task_event
        }

    def test_list_command_empty_state(self, runner, temp_data_dir):
        """Test list command with no data."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            mock_store = EventStore(temp_data_dir)
            mock_store_class.return_value = mock_store

            result = runner.invoke(app, ["list"])

            assert result.exit_code == 0
            assert "📋 Items & Tasks" in result.stdout
            assert "No active items or tasks" in result.stdout

    def test_list_command_with_inbox_items(self, runner, temp_data_dir):
        """Test list command with inbox items."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            result = runner.invoke(app, ["list"])

            assert result.exit_code == 0
            assert "📋 Items & Tasks" in result.stdout
            assert "📥 Unprocessed Inbox Items" in result.stdout
            assert "Second inbox item" in result.stdout  # Only unprocessed items
            assert "✅ Active Tasks" in result.stdout
            assert "Test Task" in result.stdout

    def test_list_command_with_all_flag(self, runner, temp_data_dir):
        """Test list command with --all flag to show completed tasks."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)

            # Complete the task
            from sidecar_os.core.sidecar_core.events import TaskCompletedEvent
            completion_event = TaskCompletedEvent(payload={
                "task_id": "test_task_1",
                "completion_note": "Test completion"
            })
            store.append(completion_event)

            mock_store_class.return_value = store

            result = runner.invoke(app, ["list", "--all"])

            assert result.exit_code == 0
            assert "🏁 Completed Tasks" in result.stdout
            assert "Test Task" in result.stdout

    def test_task_command_creates_task_from_inbox(self, runner, temp_data_dir):
        """Test task command creates task from inbox item."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            # Use partial ID of unprocessed inbox item
            inbox2_id = events["inbox2"].event_id[:8]
            result = runner.invoke(app, ["task", inbox2_id])

            assert result.exit_code == 0
            assert "✓ Created task: Second inbox item" in result.stdout
            assert "Task ID:" in result.stdout

            # Verify event was created
            all_events = store.read_all()
            task_events = [e for e in all_events if e.event_type == "task_created"]
            assert len(task_events) == 2  # Original + new one

    def test_task_command_with_custom_title(self, runner, temp_data_dir):
        """Test task command with custom title."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            inbox2_id = events["inbox2"].event_id[:8]
            result = runner.invoke(app, ["task", inbox2_id, "--title", "Custom Task Title"])

            assert result.exit_code == 0
            assert "✓ Created task: Custom Task Title" in result.stdout

    def test_task_command_nonexistent_inbox_item(self, runner, temp_data_dir):
        """Test task command with non-existent inbox item ID."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            result = runner.invoke(app, ["task", "nonexistent"])

            assert result.exit_code == 0
            assert "❌ Inbox item not found: nonexistent" in result.stdout

    def test_task_command_already_processed_item(self, runner, temp_data_dir):
        """Test task command with already processed inbox item."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            # The first inbox item is already processed (task was created from it)
            inbox1_id = events["inbox1"].event_id[:8]
            result = runner.invoke(app, ["task", inbox1_id])

            assert result.exit_code == 0
            assert "⚠️  Item already processed" in result.stdout

    def test_done_command_by_task_id(self, runner, temp_data_dir):
        """Test done command using exact task ID."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            result = runner.invoke(app, ["done", "test_task_1"])

            assert result.exit_code == 0
            assert "✓ Completed task: Test Task" in result.stdout
            assert "Task ID: test_task_1" in result.stdout

            # Verify completion event was created
            all_events = store.read_all()
            completion_events = [e for e in all_events if e.event_type == "task_completed"]
            assert len(completion_events) == 1

    def test_done_command_by_partial_id(self, runner, temp_data_dir):
        """Test done command using partial task ID."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            result = runner.invoke(app, ["done", "test_task"])

            assert result.exit_code == 0
            assert "✓ Completed task: Test Task" in result.stdout

    def test_done_command_by_title_search(self, runner, temp_data_dir):
        """Test done command using title search."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            result = runner.invoke(app, ["done", "Test"])

            assert result.exit_code == 0
            assert "✓ Completed task: Test Task" in result.stdout

    def test_done_command_nonexistent_task(self, runner, temp_data_dir):
        """Test done command with non-existent task."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)
            mock_store_class.return_value = store

            result = runner.invoke(app, ["done", "nonexistent"])

            assert result.exit_code == 0
            assert "❌ Task not found: nonexistent" in result.stdout
            assert "Try: sidecar list" in result.stdout

    def test_done_command_already_completed_task(self, runner, temp_data_dir):
        """Test done command with already completed task."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            store, events = self._create_store_with_data(temp_data_dir)

            # Complete the task first
            from sidecar_os.core.sidecar_core.events import TaskCompletedEvent
            completion_event = TaskCompletedEvent(payload={
                "task_id": "test_task_1",
                "completion_note": "Already completed"
            })
            store.append(completion_event)

            mock_store_class.return_value = store

            result = runner.invoke(app, ["done", "test_task_1"])

            assert result.exit_code == 0
            assert "⚠️  Task already completed: Test Task" in result.stdout

    def test_full_workflow_integration(self, runner, temp_data_dir):
        """Test complete workflow: add -> list -> task -> list -> done -> list --all."""
        with patch('sidecar_os.cli.commands.core.EventStore') as mock_store_class:
            def create_store():
                return EventStore(temp_data_dir)
            mock_store_class.side_effect = create_store

            # Add inbox item
            result1 = runner.invoke(app, ["add", "Complete integration testing"])
            assert result1.exit_code == 0

            # List to see inbox item
            result2 = runner.invoke(app, ["list"])
            assert result2.exit_code == 0
            assert "Complete integration testing" in result2.stdout
            assert "📥 Unprocessed Inbox Items" in result2.stdout

            # Extract inbox ID from the table output
            # Look for lines containing "..." which indicates an ID
            lines = result2.stdout.split('\n')
            inbox_id = None
            for line in lines:
                if "..." in line and not "─" in line and not "━" in line:  # Skip table borders
                    # Extract the first word which should be the ID
                    parts = line.split()
                    for part in parts:
                        if "..." in part:
                            inbox_id = part.replace("...", "").strip("│").strip()
                            break
                    if inbox_id:
                        break

            assert inbox_id is not None, f"Could not find inbox ID in output: {result2.stdout}"

            # Create task from inbox item
            result3 = runner.invoke(app, ["task", inbox_id])
            assert result3.exit_code == 0
            assert "✓ Created task: Complete integration testing" in result3.stdout

            # List to see active task
            result4 = runner.invoke(app, ["list"])
            assert result4.exit_code == 0
            assert "✅ Active Tasks" in result4.stdout
            assert "Complete" in result4.stdout
            assert "integration" in result4.stdout

            # Complete the task
            result5 = runner.invoke(app, ["done", "integration"])
            assert result5.exit_code == 0
            assert "✓ Completed task: Complete integration testing" in result5.stdout

            # List all to see completed task
            result6 = runner.invoke(app, ["list", "--all"])
            assert result6.exit_code == 0
            assert "🏁 Completed Tasks" in result6.stdout
            assert "Complete" in result6.stdout
            assert "integration" in result6.stdout