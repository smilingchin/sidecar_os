"""Artifact store implementation with JSONL persistence."""

import json
from pathlib import Path
from typing import Iterator, List

from ..events.schemas import BaseEvent


class ArtifactStore:
    """Append-only artifact store with JSONL persistence, parallel to EventStore."""

    def __init__(self, data_dir: str = "data"):
        """Initialize artifact store.

        Args:
            data_dir: Directory for storing artifact logs (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.artifact_log_path = self.data_dir / "artifacts.log"

    def register_artifact(self, event: BaseEvent) -> str:
        """Register new artifact event, returns artifact_id.

        Args:
            event: Artifact event to append (ArtifactRegisteredEvent, etc.)

        Returns:
            The event ID of the appended event
        """
        # Convert to JSON line
        event_json = event.model_dump_json()

        # Atomic write using temp file + rename (same pattern as EventStore)
        temp_path = self.artifact_log_path.with_suffix(".log.tmp")

        with open(temp_path, "w") as temp_file:
            # Copy existing content if file exists
            if self.artifact_log_path.exists():
                with open(self.artifact_log_path, "r") as existing_file:
                    temp_file.write(existing_file.read())

            # Append new artifact event
            temp_file.write(event_json + "\n")

        # Atomic rename
        temp_path.rename(self.artifact_log_path)

        return event.event_id

    def read_all_artifact_events(self) -> List[BaseEvent]:
        """Read all artifact events from the store.

        Returns:
            List of all artifact events in chronological order
        """
        if not self.artifact_log_path.exists():
            return []

        events = []
        with open(self.artifact_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    event_data = json.loads(line)
                    # Parse based on event_type
                    event = self._parse_artifact_event(event_data)
                    events.append(event)

        return events

    def read_artifact_events(self) -> Iterator[BaseEvent]:
        """Stream artifact events from the store.

        Yields:
            Artifact events in chronological order
        """
        if not self.artifact_log_path.exists():
            return

        with open(self.artifact_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    event_data = json.loads(line)
                    event = self._parse_artifact_event(event_data)
                    yield event

    def _parse_artifact_event(self, event_data: dict) -> BaseEvent:
        """Parse artifact event data into appropriate event type.

        Args:
            event_data: Raw event data from JSON

        Returns:
            Parsed event object
        """
        from ..events.schemas import (
            ArtifactRegisteredEvent,
            ArtifactLinkedEvent,
            ArtifactUnlinkedEvent,
            ArtifactArchivedEvent
        )

        event_type = event_data.get("event_type")

        if event_type == "artifact_registered":
            return ArtifactRegisteredEvent(**event_data)
        elif event_type == "artifact_linked":
            return ArtifactLinkedEvent(**event_data)
        elif event_type == "artifact_unlinked":
            return ArtifactUnlinkedEvent(**event_data)
        elif event_type == "artifact_archived":
            return ArtifactArchivedEvent(**event_data)
        else:
            # Fallback to BaseEvent for unknown types
            return BaseEvent(**event_data)

    def count_artifacts(self) -> int:
        """Count total number of artifact events in the store.

        Returns:
            Total artifact event count
        """
        if not self.artifact_log_path.exists():
            return 0

        count = 0
        with open(self.artifact_log_path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count