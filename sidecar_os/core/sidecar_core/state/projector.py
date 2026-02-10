"""State projection logic for replaying events."""

from datetime import datetime
from typing import List

from ..events import BaseEvent
from .models import SidecarState, InboxItem, Task, Project, ClarificationRequest, SystemStats


class StateProjector:
    """Projects events into current system state."""

    def project_state(self, events: List[BaseEvent]) -> SidecarState:
        """Project a list of events into the current state.

        Args:
            events: List of events in chronological order

        Returns:
            Current derived state
        """
        state = SidecarState()

        for event in events:
            self._apply_event(state, event)

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


def project_events_to_state(events: List[BaseEvent]) -> SidecarState:
    """Convenience function to project events to state.

    Args:
        events: List of events to project

    Returns:
        Derived state
    """
    projector = StateProjector()
    return projector.project_state(events)