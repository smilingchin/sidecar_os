"""State projection logic for replaying events."""

from datetime import datetime
from typing import List, Optional

from ..events import BaseEvent
from .models import SidecarState, InboxItem, Task, Project, ClarificationRequest, SystemStats, Artifact


class StateProjector:
    """Projects events into current system state."""

    def project_state(self, events: List[BaseEvent], artifact_events: Optional[List[BaseEvent]] = None) -> SidecarState:
        """Project a list of events into the current state.

        Args:
            events: List of events in chronological order
            artifact_events: Optional list of artifact events

        Returns:
            Current derived state
        """
        state = SidecarState()

        # Project main events (tasks, projects, etc.)
        for event in events:
            self._apply_event(state, event)

        # Project artifact events if provided
        if artifact_events:
            for event in artifact_events:
                self._apply_artifact_event(state, event)

        # Update final statistics
        self._update_stats(state)

        return state

    def _apply_event(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply a single event to the state.

        Args:
            state: Current state to modify
            event: Event to apply
        """
        if event.event_type == "inbox_captured":
            self._apply_inbox_captured(state, event)
        elif event.event_type == "task_created":
            self._apply_task_created(state, event)
        elif event.event_type == "task_completed":
            self._apply_task_completed(state, event)
        elif event.event_type == "task_scheduled":
            self._apply_task_scheduled(state, event)
        elif event.event_type == "task_duration_set":
            self._apply_task_duration_set(state, event)
        elif event.event_type == "task_priority_updated":
            self._apply_task_priority_updated(state, event)
        elif event.event_type == "task_status_updated":
            self._apply_task_status_updated(state, event)
        elif event.event_type == "task_project_associated":
            self._apply_task_project_associated(state, event)
        elif event.event_type == "project_created":
            self._apply_project_created(state, event)
        elif event.event_type == "project_focused":
            self._apply_project_focused(state, event)
        elif event.event_type == "project_focus_cleared":
            self._apply_project_focus_cleared(state, event)
        elif event.event_type == "clarification_requested":
            self._apply_clarification_requested(state, event)
        elif event.event_type == "clarification_resolved":
            self._apply_clarification_resolved(state, event)

        # Update tracking
        state.last_event_processed = event.event_id
        state.stats.total_events += 1
        state.stats.last_activity = event.timestamp

    def _apply_inbox_captured(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply inbox_captured event."""
        payload = event.payload

        inbox_item = InboxItem(
            event_id=event.event_id,
            text=payload.get("text", ""),
            timestamp=event.timestamp,
            priority=payload.get("priority"),
            tags=payload.get("tags", []),
            processed=False
        )

        state.inbox_items[event.event_id] = inbox_item

    def _apply_task_created(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_created event."""
        payload = event.payload

        task = Task(
            task_id=payload.get("task_id", event.event_id),
            title=payload.get("title", ""),
            description=payload.get("description"),
            created_from_event=payload.get("created_from_event", ""),
            created_at=event.timestamp,
            priority=payload.get("priority"),
            tags=payload.get("tags", []),
            status="pending",
            project_id=payload.get("project_id")
        )

        state.tasks[task.task_id] = task

        # Mark corresponding inbox item as processed if it exists
        source_event_id = payload.get("created_from_event")
        if source_event_id and source_event_id in state.inbox_items:
            state.inbox_items[source_event_id].processed = True

    def _apply_task_completed(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_completed event."""
        payload = event.payload
        task_id = payload.get("task_id")

        if task_id and task_id in state.tasks:
            task = state.tasks[task_id]
            task.status = "completed"
            task.completed_at = event.timestamp

    def _apply_task_scheduled(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_scheduled event."""
        payload = event.payload
        task_id = payload.get("task_id")
        scheduled_for_str = payload.get("scheduled_for")

        if task_id and task_id in state.tasks and scheduled_for_str:
            task = state.tasks[task_id]
            # Parse scheduled_for if it's a string
            if isinstance(scheduled_for_str, str):
                try:
                    scheduled_for = datetime.fromisoformat(scheduled_for_str)
                    task.scheduled_for = scheduled_for
                except ValueError:
                    # Skip if invalid date format
                    pass
            elif isinstance(scheduled_for_str, datetime):
                task.scheduled_for = scheduled_for_str

    def _apply_task_duration_set(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_duration_set event."""
        payload = event.payload
        task_id = payload.get("task_id")
        duration_minutes = payload.get("duration_minutes")

        if task_id and task_id in state.tasks and duration_minutes is not None:
            task = state.tasks[task_id]
            try:
                task.duration_minutes = int(duration_minutes)
            except (ValueError, TypeError):
                # Skip if invalid duration format
                pass

    def _apply_task_priority_updated(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_priority_updated event."""
        payload = event.payload
        task_id = payload.get("task_id")
        priority = payload.get("priority")

        if task_id and task_id in state.tasks and priority:
            task = state.tasks[task_id]
            # Validate priority level
            valid_priorities = ["low", "normal", "high", "urgent"]
            if priority.lower() in valid_priorities:
                task.priority = priority.lower()

    def _apply_task_status_updated(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_status_updated event."""
        payload = event.payload
        task_id = payload.get("task_id")
        status = payload.get("status")

        if task_id and task_id in state.tasks and status:
            task = state.tasks[task_id]
            # Validate status
            valid_statuses = ["pending", "in_progress", "completed", "cancelled", "on_hold"]
            if status.lower() in valid_statuses:
                task.status = status.lower()
                # If marking as completed, set completed_at timestamp if not already set
                if status.lower() == "completed" and not task.completed_at:
                    task.completed_at = event.timestamp

    def _apply_task_project_associated(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply task_project_associated event."""
        payload = event.payload
        task_id = payload.get("task_id")
        project_id = payload.get("project_id")

        if task_id and project_id and task_id in state.tasks:
            task = state.tasks[task_id]
            task.project_id = project_id

    def _apply_project_created(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply project_created event."""
        payload = event.payload

        project = Project(
            project_id=payload.get("project_id", event.event_id),
            name=payload.get("name", ""),
            description=payload.get("description"),
            aliases=payload.get("aliases", []),
            created_at=event.timestamp,
            focus_count=0,
            last_focused_at=None
        )

        state.projects[project.project_id] = project

    def _apply_project_focused(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply project_focused event."""
        payload = event.payload
        project_id = payload.get("project_id")

        if project_id and project_id in state.projects:
            project = state.projects[project_id]
            project.focus_count += 1
            project.last_focused_at = event.timestamp
            state.current_focus_project = project_id

    def _apply_project_focus_cleared(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply project_focus_cleared event."""
        state.current_focus_project = None

    def _apply_clarification_requested(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply clarification_requested event."""
        payload = event.payload

        clarification = ClarificationRequest(
            request_id=event.event_id,
            source_event_id=payload.get("source_event_id", ""),
            questions=payload.get("questions", []),
            resolved=False,
            created_at=event.timestamp
        )

        state.clarifications[clarification.request_id] = clarification

    def _apply_clarification_resolved(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply clarification_resolved event."""
        payload = event.payload
        request_id = payload.get("clarification_id", "")

        if request_id in state.clarifications:
            state.clarifications[request_id].resolved = True

    def _apply_artifact_event(self, state: SidecarState, event: BaseEvent) -> None:
        """Route artifact events to appropriate handlers."""
        if event.event_type == "artifact_registered":
            self._apply_artifact_registered(state, event)
        elif event.event_type == "artifact_linked":
            self._apply_artifact_linked(state, event)
        elif event.event_type == "artifact_unlinked":
            self._apply_artifact_unlinked(state, event)
        elif event.event_type == "artifact_archived":
            self._apply_artifact_archived(state, event)

    def _apply_artifact_registered(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply artifact_registered event."""
        payload = event.payload
        artifact = Artifact(
            artifact_id=payload.get("artifact_id", event.event_id),
            artifact_type=payload.get("artifact_type"),
            title=payload.get("title", ""),
            content=payload.get("content"),
            url=payload.get("url"),
            source=payload.get("source"),
            created_at=event.timestamp,
            created_by=payload.get("created_by"),
            task_id=payload.get("task_id"),
            project_id=payload.get("project_id"),
            metadata=payload.get("metadata", {})
        )
        state.artifacts[artifact.artifact_id] = artifact

    def _apply_artifact_linked(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply artifact_linked event."""
        payload = event.payload
        artifact_id = payload.get("artifact_id")
        task_id = payload.get("task_id")
        project_id = payload.get("project_id")

        if artifact_id and artifact_id in state.artifacts:
            if task_id:
                state.artifacts[artifact_id].task_id = task_id
            if project_id:
                state.artifacts[artifact_id].project_id = project_id

    def _apply_artifact_unlinked(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply artifact_unlinked event."""
        payload = event.payload
        artifact_id = payload.get("artifact_id")
        task_id = payload.get("task_id")
        project_id = payload.get("project_id")

        if artifact_id and artifact_id in state.artifacts:
            artifact = state.artifacts[artifact_id]
            if task_id and artifact.task_id == task_id:
                artifact.task_id = None
            if project_id and artifact.project_id == project_id:
                artifact.project_id = None

    def _apply_artifact_archived(self, state: SidecarState, event: BaseEvent) -> None:
        """Apply artifact_archived event."""
        payload = event.payload
        artifact_id = payload.get("artifact_id")

        if artifact_id and artifact_id in state.artifacts:
            state.artifacts[artifact_id].archived_at = event.timestamp

    def _update_stats(self, state: SidecarState) -> None:
        """Update derived statistics in the state.

        Args:
            state: State to update
        """
        stats = SystemStats()

        # Count totals
        stats.total_events = state.stats.total_events
        stats.inbox_count = len(state.inbox_items)
        stats.unprocessed_inbox_count = len(state.get_unprocessed_inbox())
        stats.project_count = len(state.projects)
        stats.pending_clarifications = len(state.get_pending_clarifications())

        # Count tasks by status
        active_tasks = 0
        completed_tasks = 0

        for task in state.tasks.values():
            if task.status == "completed":
                completed_tasks += 1
            else:
                active_tasks += 1

        stats.active_tasks = active_tasks
        stats.completed_tasks = completed_tasks
        stats.last_activity = state.stats.last_activity

        state.stats = stats


def project_events_to_state(events: List[BaseEvent], artifact_events: Optional[List[BaseEvent]] = None) -> SidecarState:
    """Convenience function to project events to state.

    Args:
        events: List of events to project
        artifact_events: Optional list of artifact events

    Returns:
        Derived state
    """
    projector = StateProjector()
    return projector.project_state(events, artifact_events)