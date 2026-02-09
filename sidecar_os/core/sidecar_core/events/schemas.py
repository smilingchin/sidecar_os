"""Event schema definitions for Sidecar OS."""

from datetime import datetime, UTC
from typing import Any, Dict, Literal
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


# Type alias for all event types
Event = InboxCapturedEvent | TaskCreatedEvent | TaskCompletedEvent | TaskScheduledEvent