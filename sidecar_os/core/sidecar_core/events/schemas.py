"""Event schema definitions for Sidecar OS."""

from datetime import datetime, UTC
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    """Base event schema with common fields."""

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    source: str = "cli"
    payload: Dict[str, Any]


class InboxCapturedEvent(BaseEvent):
    """Event for capturing inbox items (tasks, notes, ideas)."""

    event_type: Literal["inbox_captured"] = "inbox_captured"


class TaskCreatedEvent(BaseEvent):
    """Event for creating structured tasks from inbox items."""

    event_type: Literal["task_created"] = "task_created"


class TaskCompletedEvent(BaseEvent):
    """Event for marking tasks as completed."""

    event_type: Literal["task_completed"] = "task_completed"


class TaskScheduledEvent(BaseEvent):
    """Event for scheduling tasks for specific dates."""

    event_type: Literal["task_scheduled"] = "task_scheduled"


class ProjectCreatedEvent(BaseEvent):
    """Event for creating new projects."""

    event_type: Literal["project_created"] = "project_created"


class ProjectFocusedEvent(BaseEvent):
    """Event for setting current project focus."""

    event_type: Literal["project_focused"] = "project_focused"


class ProjectFocusClearedEvent(BaseEvent):
    """Event for clearing current project focus."""

    event_type: Literal["project_focus_cleared"] = "project_focus_cleared"


class ClarificationRequestedEvent(BaseEvent):
    """Event for requesting clarification from user."""

    event_type: Literal["clarification_requested"] = "clarification_requested"


class ClarificationResolvedEvent(BaseEvent):
    """Event for resolving a clarification with user answers."""

    event_type: Literal["clarification_resolved"] = "clarification_resolved"


# Type alias for all event types
Event = (
    InboxCapturedEvent |
    TaskCreatedEvent |
    TaskCompletedEvent |
    TaskScheduledEvent |
    ProjectCreatedEvent |
    ProjectFocusedEvent |
    ProjectFocusClearedEvent |
    ClarificationRequestedEvent |
    ClarificationResolvedEvent
)