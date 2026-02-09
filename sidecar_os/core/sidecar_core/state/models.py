"""State models for Sidecar OS."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class InboxItem(BaseModel):
    """An item captured in the inbox."""

    event_id: str
    text: str
    timestamp: datetime
    priority: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    processed: bool = False


class Task(BaseModel):
    """A structured task derived from inbox items."""

    task_id: str
    title: str
    description: Optional[str] = None
    created_from_event: str  # Reference to originating event
    created_at: datetime
    completed_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = None
    priority: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, cancelled


class SystemStats(BaseModel):
    """Statistics about the system state."""

    total_events: int = 0
    inbox_count: int = 0
    unprocessed_inbox_count: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    last_activity: Optional[datetime] = None


class SidecarState(BaseModel):
    """Complete derived state of the Sidecar OS system."""

    inbox_items: Dict[str, InboxItem] = Field(default_factory=dict)
    tasks: Dict[str, Task] = Field(default_factory=dict)
    stats: SystemStats = Field(default_factory=SystemStats)
    last_event_processed: Optional[str] = None

    def get_recent_inbox_items(self, limit: int = 10) -> List[InboxItem]:
        """Get recent inbox items sorted by timestamp."""
        items = sorted(
            self.inbox_items.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return items[:limit]

    def get_active_tasks(self) -> List[Task]:
        """Get all active (non-completed) tasks."""
        return [
            task for task in self.tasks.values()
            if task.status != "completed"
        ]

    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [
            task for task in self.tasks.values()
            if task.status == "completed"
        ]

    def get_unprocessed_inbox(self) -> List[InboxItem]:
        """Get inbox items that haven't been processed into tasks."""
        return [
            item for item in self.inbox_items.values()
            if not item.processed
        ]