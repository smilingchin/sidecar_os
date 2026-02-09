"""Tests for event system components."""

import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime

from sidecar_os.core.sidecar_core.events import (
    BaseEvent,
    InboxCapturedEvent,
    TaskCreatedEvent,
    EventStore,
)


class TestEventSchemas:
    """Test event schema validation and creation."""

    def test_base_event_creation(self):
        """Test BaseEvent creation with required fields."""
        event = BaseEvent(
            event_type="test_event",
            payload={"message": "test"}
        )

        assert event.event_type == "test_event"
        assert event.payload == {"message": "test"}
        assert event.source == "cli"  # default value
        assert isinstance(event.event_id, str)
        assert len(event.event_id) == 36  # UUID format
        assert isinstance(event.timestamp, datetime)

    def test_inbox_captured_event(self):
        """Test InboxCapturedEvent creation."""
        event = InboxCapturedEvent(
            payload={"text": "New inbox item", "priority": "normal"}
        )

        assert event.event_type == "inbox_captured"
        assert event.payload["text"] == "New inbox item"
        assert event.payload["priority"] == "normal"

    def test_task_created_event(self):
        """Test TaskCreatedEvent creation."""
        event = TaskCreatedEvent(
            payload={"title": "Test Task", "description": "A test task"}
        )

        assert event.event_type == "task_created"
        assert event.payload["title"] == "Test Task"
        assert event.payload["description"] == "A test task"

    def test_event_json_serialization(self):
        """Test that events can be serialized to JSON."""
        event = InboxCapturedEvent(
            payload={"text": "Test item"}
        )

        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["event_type"] == "inbox_captured"
        assert data["payload"]["text"] == "Test item"
        assert "event_id" in data
        assert "timestamp" in data


class TestEventStore:
    """Test event store operations."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary event store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield EventStore(temp_dir)

    def test_append_single_event(self, temp_store):
        """Test appending a single event."""
        event = InboxCapturedEvent(
            payload={"text": "Test inbox item"}
        )

        event_id = temp_store.append(event)

        assert event_id == event.event_id
        assert temp_store.count() == 1

    def test_append_multiple_events(self, temp_store):
        """Test appending multiple events."""
        events = [
            InboxCapturedEvent(payload={"text": "Item 1"}),
            InboxCapturedEvent(payload={"text": "Item 2"}),
            TaskCreatedEvent(payload={"title": "Task 1"}),
        ]

        for event in events:
            temp_store.append(event)

        assert temp_store.count() == 3

    def test_read_all_events(self, temp_store):
        """Test reading all events from store."""
        original_events = [
            InboxCapturedEvent(payload={"text": "Item 1"}),
            InboxCapturedEvent(payload={"text": "Item 2"}),
        ]

        for event in original_events:
            temp_store.append(event)

        read_events = temp_store.read_all()

        assert len(read_events) == 2
        assert read_events[0].payload["text"] == "Item 1"
        assert read_events[1].payload["text"] == "Item 2"

    def test_read_events_stream(self, temp_store):
        """Test streaming events from store."""
        original_events = [
            InboxCapturedEvent(payload={"text": "Item 1"}),
            TaskCreatedEvent(payload={"title": "Task 1"}),
        ]

        for event in original_events:
            temp_store.append(event)

        streamed_events = list(temp_store.read_events())

        assert len(streamed_events) == 2
        assert streamed_events[0].event_type == "inbox_captured"
        assert streamed_events[1].event_type == "task_created"

    def test_empty_store(self, temp_store):
        """Test operations on empty store."""
        assert temp_store.count() == 0
        assert temp_store.read_all() == []
        assert list(temp_store.read_events()) == []

    def test_event_persistence(self, temp_store):
        """Test that events persist across store instances."""
        # Create and append event
        event = InboxCapturedEvent(payload={"text": "Persistent item"})
        temp_store.append(event)

        # Create new store instance pointing to same directory
        new_store = EventStore(str(temp_store.data_dir))
        read_events = new_store.read_all()

        assert len(read_events) == 1
        assert read_events[0].payload["text"] == "Persistent item"

    def test_atomic_write_operation(self, temp_store):
        """Test that append operations are atomic."""
        # Verify file doesn't exist initially
        assert not temp_store.event_log_path.exists()

        # Append event
        event = InboxCapturedEvent(payload={"text": "Atomic test"})
        temp_store.append(event)

        # Verify file exists and contains proper content
        assert temp_store.event_log_path.exists()

        # Read file directly to verify JSONL format
        with open(temp_store.event_log_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        event_data = json.loads(lines[0].strip())
        assert event_data["event_type"] == "inbox_captured"
        assert event_data["payload"]["text"] == "Atomic test"

    def test_event_ordering(self, temp_store):
        """Test that events maintain chronological order."""
        events = []
        for i in range(5):
            event = InboxCapturedEvent(payload={"text": f"Item {i}"})
            events.append(event)
            temp_store.append(event)

        read_events = temp_store.read_all()

        # Verify same order
        for i, read_event in enumerate(read_events):
            assert read_event.payload["text"] == f"Item {i}"
            assert read_event.event_id == events[i].event_id