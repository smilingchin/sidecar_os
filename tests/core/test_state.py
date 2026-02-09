"""Tests for state projection system."""

from datetime import datetime, UTC
import pytest

from sidecar_os.core.sidecar_core.events import (
    InboxCapturedEvent,
    TaskCreatedEvent,
    TaskCompletedEvent,
    TaskScheduledEvent,
)
from sidecar_os.core.sidecar_core.state import (
    SidecarState,
    InboxItem,
    Task,
    StateProjector,
    project_events_to_state,
)


class TestStateModels:
    """Test state model functionality."""

    def test_sidecar_state_creation(self):
        """Test SidecarState creation and default values."""
        state = SidecarState()

        assert len(state.inbox_items) == 0
        assert len(state.tasks) == 0
        assert state.stats.total_events == 0
        assert state.stats.inbox_count == 0
        assert state.last_event_processed is None

    def test_get_recent_inbox_items(self):
        """Test getting recent inbox items sorted by timestamp."""
        state = SidecarState()

        # Add inbox items with different timestamps
        now = datetime.now(UTC)
        state.inbox_items["1"] = InboxItem(
            event_id="1",
            text="Item 1",
            timestamp=now
        )
        state.inbox_items["2"] = InboxItem(
            event_id="2",
            text="Item 2",
            timestamp=now.replace(hour=now.hour - 1)
        )

        recent = state.get_recent_inbox_items(limit=1)
        assert len(recent) == 1
        assert recent[0].text == "Item 1"  # Most recent

    def test_get_active_tasks(self):
        """Test getting active (non-completed) tasks."""
        state = SidecarState()

        state.tasks["1"] = Task(
            task_id="1",
            title="Active Task",
            created_from_event="event_1",
            created_at=datetime.now(UTC),
            status="pending"
        )
        state.tasks["2"] = Task(
            task_id="2",
            title="Completed Task",
            created_from_event="event_2",
            created_at=datetime.now(UTC),
            status="completed"
        )

        active = state.get_active_tasks()
        assert len(active) == 1
        assert active[0].title == "Active Task"

    def test_get_unprocessed_inbox(self):
        """Test getting unprocessed inbox items."""
        state = SidecarState()

        state.inbox_items["1"] = InboxItem(
            event_id="1",
            text="Unprocessed",
            timestamp=datetime.now(UTC),
            processed=False
        )
        state.inbox_items["2"] = InboxItem(
            event_id="2",
            text="Processed",
            timestamp=datetime.now(UTC),
            processed=True
        )

        unprocessed = state.get_unprocessed_inbox()
        assert len(unprocessed) == 1
        assert unprocessed[0].text == "Unprocessed"


class TestStateProjector:
    """Test state projection logic."""

    @pytest.fixture
    def projector(self):
        """Create a StateProjector instance."""
        return StateProjector()

    def test_empty_event_list(self, projector):
        """Test projecting empty event list."""
        state = projector.project_state([])

        assert len(state.inbox_items) == 0
        assert len(state.tasks) == 0
        assert state.stats.total_events == 0

    def test_inbox_captured_event(self, projector):
        """Test projecting inbox_captured event."""
        event = InboxCapturedEvent(
            payload={
                "text": "Test inbox item",
                "priority": "high",
                "tags": ["work", "urgent"]
            }
        )

        state = projector.project_state([event])

        assert len(state.inbox_items) == 1
        assert state.stats.total_events == 1
        assert state.stats.inbox_count == 1
        assert state.stats.unprocessed_inbox_count == 1

        inbox_item = list(state.inbox_items.values())[0]
        assert inbox_item.text == "Test inbox item"
        assert inbox_item.priority == "high"
        assert inbox_item.tags == ["work", "urgent"]
        assert not inbox_item.processed

    def test_task_created_event(self, projector):
        """Test projecting task_created event."""
        # First create an inbox item
        inbox_event = InboxCapturedEvent(
            payload={"text": "Original inbox item"}
        )

        # Then create a task from it
        task_event = TaskCreatedEvent(
            payload={
                "task_id": "task_1",
                "title": "Test Task",
                "description": "A test task",
                "created_from_event": inbox_event.event_id,
                "priority": "medium",
                "tags": ["work"]
            }
        )

        state = projector.project_state([inbox_event, task_event])

        # Check task was created
        assert len(state.tasks) == 1
        assert state.stats.active_tasks == 1

        task = state.tasks["task_1"]
        assert task.title == "Test Task"
        assert task.description == "A test task"
        assert task.priority == "medium"
        assert task.tags == ["work"]
        assert task.status == "pending"

        # Check inbox item was marked as processed
        inbox_item = state.inbox_items[inbox_event.event_id]
        assert inbox_item.processed is True

    def test_task_completed_event(self, projector):
        """Test projecting task_completed event."""
        # Create task
        task_event = TaskCreatedEvent(
            payload={
                "task_id": "task_1",
                "title": "Test Task",
                "created_from_event": "inbox_1"
            }
        )

        # Complete task
        complete_event = TaskCompletedEvent(
            payload={"task_id": "task_1"}
        )

        state = projector.project_state([task_event, complete_event])

        # Check task status
        task = state.tasks["task_1"]
        assert task.status == "completed"
        assert task.completed_at is not None
        assert state.stats.completed_tasks == 1
        assert state.stats.active_tasks == 0

    def test_task_scheduled_event(self, projector):
        """Test projecting task_scheduled event."""
        # Create task
        task_event = TaskCreatedEvent(
            payload={
                "task_id": "task_1",
                "title": "Test Task",
                "created_from_event": "inbox_1"
            }
        )

        # Schedule task
        schedule_time = datetime.now(UTC).isoformat()
        schedule_event = TaskScheduledEvent(
            payload={
                "task_id": "task_1",
                "scheduled_for": schedule_time
            }
        )

        state = projector.project_state([task_event, schedule_event])

        # Check task scheduling
        task = state.tasks["task_1"]
        assert task.scheduled_for is not None
        assert task.scheduled_for.isoformat() == schedule_time

    def test_multiple_events_sequence(self, projector):
        """Test projecting a sequence of different events."""
        events = []

        # Create inbox items
        for i in range(3):
            events.append(InboxCapturedEvent(
                payload={"text": f"Inbox item {i}"}
            ))

        # Create tasks from some inbox items
        events.append(TaskCreatedEvent(
            payload={
                "task_id": "task_1",
                "title": "Task from inbox 0",
                "created_from_event": events[0].event_id
            }
        ))

        events.append(TaskCreatedEvent(
            payload={
                "task_id": "task_2",
                "title": "Task from inbox 1",
                "created_from_event": events[1].event_id
            }
        ))

        # Complete one task
        events.append(TaskCompletedEvent(
            payload={"task_id": "task_1"}
        ))

        state = projector.project_state(events)

        # Verify final state
        assert len(state.inbox_items) == 3
        assert len(state.tasks) == 2
        assert state.stats.total_events == 6
        assert state.stats.inbox_count == 3
        assert state.stats.unprocessed_inbox_count == 1  # inbox 2 not processed
        assert state.stats.active_tasks == 1  # task_2
        assert state.stats.completed_tasks == 1  # task_1

        # Verify processed status
        assert state.inbox_items[events[0].event_id].processed is True
        assert state.inbox_items[events[1].event_id].processed is True
        assert state.inbox_items[events[2].event_id].processed is False

    def test_stats_update(self, projector):
        """Test that statistics are correctly updated."""
        events = [
            InboxCapturedEvent(payload={"text": "Item 1"}),
            InboxCapturedEvent(payload={"text": "Item 2"}),
            TaskCreatedEvent(payload={
                "task_id": "task_1",
                "title": "Task 1",
                "created_from_event": "inbox_1"
            }),
        ]

        state = projector.project_state(events)

        assert state.stats.total_events == 3
        assert state.stats.last_activity == events[-1].timestamp
        assert state.last_event_processed == events[-1].event_id

    def test_convenience_function(self):
        """Test the convenience function for state projection."""
        events = [
            InboxCapturedEvent(payload={"text": "Test item"}),
        ]

        state = project_events_to_state(events)

        assert len(state.inbox_items) == 1
        assert state.stats.total_events == 1