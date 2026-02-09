"""Event store implementation with JSONL persistence."""

import json
import os
from pathlib import Path
from typing import Iterator, List

from .schemas import BaseEvent, Event


class EventStore:
    """Append-only event store with JSONL persistence."""

    def __init__(self, data_dir: str = "data"):
        """Initialize event store.

        Args:
            data_dir: Directory for storing event logs (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.event_log_path = self.data_dir / "events.log"

    def append(self, event: BaseEvent) -> str:
        """Append an event to the store.

        Args:
            event: Event to append

        Returns:
            The event ID of the appended event
        """
        # Convert to JSON line
        event_json = event.model_dump_json()

        # Atomic write using temp file + rename
        temp_path = self.event_log_path.with_suffix(".log.tmp")

        with open(temp_path, "w") as temp_file:
            # Copy existing content if file exists
            if self.event_log_path.exists():
                with open(self.event_log_path, "r") as existing_file:
                    temp_file.write(existing_file.read())

            # Append new event
            temp_file.write(event_json + "\n")

        # Atomic rename
        temp_path.rename(self.event_log_path)

        return event.event_id

    def read_all(self) -> List[BaseEvent]:
        """Read all events from the store.

        Returns:
            List of all events in chronological order
        """
        if not self.event_log_path.exists():
            return []

        events = []
        with open(self.event_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    event_data = json.loads(line)
                    # Parse based on event_type
                    event = self._parse_event(event_data)
                    events.append(event)

        return events

    def read_events(self) -> Iterator[BaseEvent]:
        """Stream events from the store.

        Yields:
            Events in chronological order
        """
        if not self.event_log_path.exists():
            return

        with open(self.event_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    event_data = json.loads(line)
                    event = self._parse_event(event_data)
                    yield event

    def _parse_event(self, event_data: dict) -> BaseEvent:
        """Parse event data into appropriate event type.

        Args:
            event_data: Raw event data from JSON

        Returns:
            Parsed event object
        """
        from .schemas import (
            InboxCapturedEvent,
            TaskCreatedEvent,
            TaskCompletedEvent,
            TaskScheduledEvent
        )

        event_type = event_data.get("event_type")

        if event_type == "inbox_captured":
            return InboxCapturedEvent(**event_data)
        elif event_type == "task_created":
            return TaskCreatedEvent(**event_data)
        elif event_type == "task_completed":
            return TaskCompletedEvent(**event_data)
        elif event_type == "task_scheduled":
            return TaskScheduledEvent(**event_data)
        else:
            # Fallback to BaseEvent for unknown types
            return BaseEvent(**event_data)

    def count(self) -> int:
        """Count total number of events in the store.

        Returns:
            Total event count
        """
        if not self.event_log_path.exists():
            return 0

        count = 0
        with open(self.event_log_path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count