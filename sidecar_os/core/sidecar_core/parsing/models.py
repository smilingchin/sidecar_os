"""Data models for mixed content parsing results."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParsedTask:
    """A task extracted from mixed content."""
    title: str
    description: Optional[str] = None
    priority: Optional[str] = None  # high, normal, low, urgent
    status: Optional[str] = None    # pending, in_progress, completed
    due_date: Optional[str] = None  # ISO format or relative like "tomorrow"
    duration_minutes: Optional[int] = None
    project_hints: List[str] = None  # Keywords that suggest project association
    confidence: float = 0.0

    def __post_init__(self):
        if self.project_hints is None:
            self.project_hints = []


@dataclass
class ParsedArtifact:
    """An artifact extracted from mixed content."""
    artifact_type: str  # slack_msg, email, doc, meeting_notes, call_notes, etc.
    title: str
    content: Optional[str] = None     # Full content if embedded in input
    url: Optional[str] = None         # Extracted URL if present
    source: Optional[str] = None      # Source identifier (auto-generated if not provided)
    project_hints: List[str] = None   # Keywords that suggest project association
    task_hints: List[str] = None      # Keywords that suggest task relationship
    metadata: Dict[str, Any] = None   # Additional metadata (participants, timestamps, etc.)
    confidence: float = 0.0

    def __post_init__(self):
        if self.project_hints is None:
            self.project_hints = []
        if self.task_hints is None:
            self.task_hints = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ParsedRelationship:
    """A relationship between tasks and artifacts."""
    relationship_type: str  # task_artifact_link, artifact_supports_task, task_references_artifact
    task_index: int         # Index in ParsedContent.tasks list
    artifact_index: int     # Index in ParsedContent.artifacts list
    description: str        # Human description of the relationship
    confidence: float = 0.0


@dataclass
class ParsedContent:
    """Complete result of mixed content parsing."""
    tasks: List[ParsedTask]
    artifacts: List[ParsedArtifact]
    relationships: List[ParsedRelationship]
    overall_confidence: float
    explanation: str            # LLM explanation of what was parsed
    project_suggestions: List[str] = None  # Global project hints
    parsing_method: str = "mixed_content_llm"  # How this was parsed

    def __post_init__(self):
        if self.project_suggestions is None:
            self.project_suggestions = []

    def has_tasks(self) -> bool:
        """Check if any tasks were parsed."""
        return len(self.tasks) > 0

    def has_artifacts(self) -> bool:
        """Check if any artifacts were parsed."""
        return len(self.artifacts) > 0

    def is_empty(self) -> bool:
        """Check if parsing result is empty."""
        return not self.has_tasks() and not self.has_artifacts()

    def get_linked_artifacts_for_task(self, task_index: int) -> List[int]:
        """Get artifact indices linked to a specific task."""
        return [
            rel.artifact_index for rel in self.relationships
            if rel.task_index == task_index
        ]

    def get_linked_tasks_for_artifact(self, artifact_index: int) -> List[int]:
        """Get task indices linked to a specific artifact."""
        return [
            rel.task_index for rel in self.relationships
            if rel.artifact_index == artifact_index
        ]