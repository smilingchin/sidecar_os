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
    scheduled_for: Optional[datetime] = None  # Due date
    duration_minutes: Optional[int] = None  # Estimated duration in minutes
    priority: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, cancelled
    project_id: Optional[str] = None  # Associated project


class Project(BaseModel):
    """A project that groups related tasks and activities."""

    project_id: str
    name: str
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    created_at: datetime
    focus_count: int = 0
    last_focused_at: Optional[datetime] = None


class ClarificationRequest(BaseModel):
    """A request for user clarification on ambiguous input."""

    request_id: str
    source_event_id: str
    questions: List[str]
    resolved: bool = False
    created_at: datetime


class SystemStats(BaseModel):
    """Statistics about the system state."""

    total_events: int = 0
    inbox_count: int = 0
    unprocessed_inbox_count: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    project_count: int = 0
    pending_clarifications: int = 0
    llm_requests_count: int = 0
    llm_daily_cost: float = 0.0
    last_activity: Optional[datetime] = None


class SidecarState(BaseModel):
    """Complete derived state of the Sidecar OS system."""

    inbox_items: Dict[str, InboxItem] = Field(default_factory=dict)
    tasks: Dict[str, Task] = Field(default_factory=dict)
    projects: Dict[str, Project] = Field(default_factory=dict)
    clarifications: Dict[str, ClarificationRequest] = Field(default_factory=dict)
    current_focus_project: Optional[str] = None
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

    def get_recent_projects(self, limit: int = 10) -> List[Project]:
        """Get recently active projects sorted by last focused date."""
        projects = sorted(
            self.projects.values(),
            key=lambda x: x.last_focused_at or x.created_at,
            reverse=True
        )
        return projects[:limit]

    def get_tasks_for_project(self, project_id: str) -> List[Task]:
        """Get all tasks associated with a project."""
        return [
            task for task in self.tasks.values()
            if task.project_id == project_id
        ]

    def get_pending_clarifications(self) -> List[ClarificationRequest]:
        """Get all pending clarification requests."""
        return [
            req for req in self.clarifications.values()
            if not req.resolved
        ]

    def find_project_by_alias(self, alias: str) -> Optional[Project]:
        """Find a project by name or alias (case-insensitive)."""
        alias_lower = alias.lower()
        for project in self.projects.values():
            if (project.name.lower() == alias_lower or
                any(a.lower() == alias_lower for a in project.aliases)):
                return project
        return None

    def get_tasks_due_today(self) -> List[Task]:
        """Get tasks due today."""
        from datetime import date
        today = date.today()
        return [
            task for task in self.get_active_tasks()
            if task.scheduled_for and task.scheduled_for.date() == today
        ]

    def get_overdue_tasks(self) -> List[Task]:
        """Get tasks that are overdue (past due date)."""
        from datetime import datetime
        now = datetime.now()
        return [
            task for task in self.get_active_tasks()
            if task.scheduled_for and task.scheduled_for < now
        ]

    def get_tasks_sorted_by_due_date(self) -> List[Task]:
        """Get active tasks sorted by due date (overdue first, then by proximity)."""
        from datetime import datetime
        now = datetime.now()

        def sort_key(task: Task):
            if not task.scheduled_for:
                return (2, datetime.max)  # No due date goes to end
            overdue = task.scheduled_for < now
            return (0 if overdue else 1, task.scheduled_for)

        return sorted(self.get_active_tasks(), key=sort_key)