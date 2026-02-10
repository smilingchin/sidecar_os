"""Event system for Sidecar OS."""

from .schemas import (
    BaseEvent,
    Event,
    InboxCapturedEvent,
    TaskCreatedEvent,
    TaskCompletedEvent,
    TaskScheduledEvent,
    ProjectCreatedEvent,
    ProjectFocusedEvent,
    ProjectFocusClearedEvent,
    ClarificationRequestedEvent,
    ClarificationResolvedEvent,
)
from .store import EventStore

__all__ = [
    "BaseEvent",
    "Event",
    "InboxCapturedEvent",
    "TaskCreatedEvent",
    "TaskCompletedEvent",
    "TaskScheduledEvent",
    "ProjectCreatedEvent",
    "ProjectFocusedEvent",
    "ProjectFocusClearedEvent",
    "ClarificationRequestedEvent",
    "ClarificationResolvedEvent",
    "EventStore",
]