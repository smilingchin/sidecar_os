"""Project cleanup utilities for fixing messy project data."""

from typing import Dict, List, Set
from dataclasses import dataclass

from ..events import EventStore
from ..events.schemas import ProjectCreatedEvent, BaseEvent
from ..state import project_events_to_state
from .matcher import ProjectMatcher


@dataclass
class CleanupSuggestion:
    """Suggested cleanup action for a project."""
    project_id: str
    action: str  # "delete", "rename", "merge"
    reason: str
    new_name: str = ""
    merge_into: str = ""


class ProjectCleanupManager:
    """Manages cleanup of messy project data."""

    def __init__(self):
        """Initialize cleanup manager."""
        self.project_matcher = ProjectMatcher()
        self.event_store = EventStore()

    def analyze_cleanup_opportunities(self) -> List[CleanupSuggestion]:
        """Analyze current projects and suggest cleanup actions.

        Returns:
            List of CleanupSuggestion objects
        """
        # Load current state
        events = self.event_store.read_all()
        state = project_events_to_state(events)

        suggestions = []

        for project_id, project in state.projects.items():
            # Check if project should be deleted (messy name)
            cleaned_name = self.project_matcher.cleanup_project_name(project.name)
            if cleaned_name is None:
                suggestions.append(CleanupSuggestion(
                    project_id=project_id,
                    action="delete",
                    reason=f"Project name '{project.name}' appears to be clarification text, not a real project"
                ))
                continue

            # Check if project should be renamed (normalize)
            canonical_id, display_name = self.project_matcher.normalize_project_name(project.name)
            if canonical_id != project_id or display_name != project.name:
                suggestions.append(CleanupSuggestion(
                    project_id=project_id,
                    action="rename",
                    reason=f"Normalize project name to follow standard format",
                    new_name=f"{canonical_id} ('{display_name}')"
                ))

        # Look for potential duplicates/merges
        project_names = [(pid, p.name) for pid, p in state.projects.items()]
        for i, (pid1, name1) in enumerate(project_names):
            for pid2, name2 in project_names[i+1:]:
                if self._could_be_same_project(name1, name2):
                    suggestions.append(CleanupSuggestion(
                        project_id=pid1,
                        action="merge",
                        reason=f"'{name1}' and '{name2}' might be the same project",
                        merge_into=pid2
                    ))

        return suggestions

    def _could_be_same_project(self, name1: str, name2: str) -> bool:
        """Check if two project names could refer to the same project."""
        # Normalize for comparison
        norm1 = name1.lower().replace(' ', '').replace('-', '').replace('_', '')
        norm2 = name2.lower().replace(' ', '').replace('-', '').replace('_', '')

        # Simple similarity check
        if norm1 == norm2:
            return True

        # Check if one is substring of other (with some flexibility)
        if len(norm1) >= 3 and len(norm2) >= 3:
            if norm1 in norm2 or norm2 in norm1:
                return True

        return False

    def get_cleanup_summary(self) -> str:
        """Get a human-readable summary of cleanup opportunities."""
        suggestions = self.analyze_cleanup_opportunities()

        if not suggestions:
            return "✅ No cleanup needed - all projects look good!"

        summary = [f"🧹 Found {len(suggestions)} cleanup opportunities:\n"]

        deletes = [s for s in suggestions if s.action == "delete"]
        renames = [s for s in suggestions if s.action == "rename"]
        merges = [s for s in suggestions if s.action == "merge"]

        if deletes:
            summary.append(f"📍 DELETE ({len(deletes)} projects):")
            for s in deletes[:5]:  # Show first 5
                summary.append(f"  • '{s.project_id}' - {s.reason}")
            if len(deletes) > 5:
                summary.append(f"  ... and {len(deletes) - 5} more")

        if renames:
            summary.append(f"\n🏷️  RENAME ({len(renames)} projects):")
            for s in renames[:5]:  # Show first 5
                summary.append(f"  • '{s.project_id}' → {s.new_name}")
            if len(renames) > 5:
                summary.append(f"  ... and {len(renames) - 5} more")

        if merges:
            summary.append(f"\n🔗 POTENTIAL MERGES ({len(merges)} suggestions):")
            for s in merges[:3]:  # Show first 3
                summary.append(f"  • '{s.project_id}' + '{s.merge_into}' - {s.reason}")
            if len(merges) > 3:
                summary.append(f"  ... and {len(merges) - 3} more")

        summary.append(f"\n💡 Use project cleanup commands to fix these issues automatically.")

        return "\n".join(summary)