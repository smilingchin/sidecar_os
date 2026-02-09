"""State management for Sidecar OS."""

from .models import (
    InboxItem,
    Task,
    SystemStats,
    SidecarState,
)
from .projector import StateProjector, project_events_to_state

__all__ = [
    "InboxItem",
    "Task",
    "SystemStats",
    "SidecarState",
    "StateProjector",
    "project_events_to_state",
]