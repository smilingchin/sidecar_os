"""Event log migration utilities for cleaning up messy data."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
from dataclasses import dataclass

from ..events import EventStore
from ..events.schemas import BaseEvent
from ..state import project_events_to_state
from .cleanup import ProjectCleanupManager, CleanupSuggestion


@dataclass
class MigrationResult:
    """Result of event log migration."""
    events_processed: int
    events_kept: int
    events_filtered: int
    projects_deleted: List[str]
    projects_renamed: Dict[str, str]  # old_id -> new_id
    backup_path: str


class EventLogMigrator:
    """Migrates event log by filtering out garbage data and keeping clean events."""

    def __init__(self):
        """Initialize migrator."""
        self.event_store = EventStore()
        self.cleanup_manager = ProjectCleanupManager()

    def migrate_to_clean_log(self, dry_run: bool = True) -> MigrationResult:
        """Migrate current event log to clean version.

        Args:
            dry_run: If True, only analyze what would be done without making changes

        Returns:
            MigrationResult with details of what was done
        """
        # Load current events
        events = self.event_store.read_all()

        if not events:
            return MigrationResult(
                events_processed=0, events_kept=0, events_filtered=0,
                projects_deleted=[], projects_renamed={},
                backup_path=""
            )

        # Get cleanup suggestions
        suggestions = self.cleanup_manager.analyze_cleanup_opportunities()

        # Identify projects to delete and rename
        projects_to_delete = set()
        projects_to_rename = {}

        for suggestion in suggestions:
            if suggestion.action == "delete":
                projects_to_delete.add(suggestion.project_id)
            elif suggestion.action == "rename":
                # Extract new name from suggestion.new_name format "new_id ('Display Name')"
                new_id = suggestion.new_name.split()[0]
                projects_to_rename[suggestion.project_id] = new_id

        # Filter events
        clean_events = []
        filtered_count = 0

        for event in events:
            should_keep = self._should_keep_event(event, projects_to_delete, projects_to_rename)

            if should_keep:
                # Apply project renames if needed
                updated_event = self._update_event_project_references(event, projects_to_rename)
                clean_events.append(updated_event)
            else:
                filtered_count += 1

        if not dry_run:
            # Create backup
            backup_path = self._create_backup()

            # Write clean events to new log
            self._write_clean_events(clean_events)
        else:
            backup_path = "dry-run-no-backup"

        return MigrationResult(
            events_processed=len(events),
            events_kept=len(clean_events),
            events_filtered=filtered_count,
            projects_deleted=list(projects_to_delete),
            projects_renamed=projects_to_rename,
            backup_path=backup_path
        )

    def _should_keep_event(self, event: BaseEvent, projects_to_delete: Set[str], projects_to_rename: Dict[str, str]) -> bool:
        """Determine if an event should be kept in the clean log."""

        # Always keep inbox events (they're the original user input)
        if event.event_type == "inbox_captured":
            return True

        # Check if event is related to a project that should be deleted
        project_id = None
        if hasattr(event, 'payload') and event.payload:
            project_id = event.payload.get('project_id')

        if project_id and project_id in projects_to_delete:
            return False

        # Keep events related to projects that will be renamed (we'll update them)
        if project_id and project_id in projects_to_rename:
            return True

        # Filter out some specific garbage patterns in event data
        if event.event_type == "project_created":
            if event.payload and event.payload.get('name'):
                project_name = event.payload['name'].lower()
                garbage_patterns = [
                    'yes need to complete',
                    'status update',
                    'does this relate',
                    'task to',
                    'relate to'
                ]
                for pattern in garbage_patterns:
                    if pattern in project_name:
                        return False

        # Filter out bad tasks (those with garbage titles)
        if event.event_type == "task_created":
            if event.payload and event.payload.get('title'):
                title = event.payload['title'].lower()
                if title.startswith('yes need to complete') or title == 'task':
                    return False

        # Filter out clarification events related to garbage projects
        if event.event_type == "clarification_requested":
            if event.payload and event.payload.get('original_text'):
                text = event.payload['original_text'].lower()
                if 'yes need to complete' in text or 'does this relate' in text:
                    return False

        # Keep everything else
        return True

    def _update_event_project_references(self, event: BaseEvent, projects_to_rename: Dict[str, str]) -> BaseEvent:
        """Update project references in event payload if needed."""
        if not (hasattr(event, 'payload') and event.payload and projects_to_rename):
            return event

        # Create a copy of the event with updated payload
        updated_payload = event.payload.copy()

        # Update project_id references
        if 'project_id' in updated_payload:
            old_id = updated_payload['project_id']
            if old_id in projects_to_rename:
                updated_payload['project_id'] = projects_to_rename[old_id]

        # Update project names for project_created events
        if event.event_type == "project_created" and 'name' in updated_payload:
            old_id = updated_payload.get('project_id')
            if old_id and old_id in projects_to_rename:
                # Update the display name to proper format
                new_id = projects_to_rename[old_id]
                display_name = ' '.join(word.capitalize() for word in new_id.split('_'))
                updated_payload['name'] = display_name

        # Create new event with updated payload
        # This is a bit hacky but works for our purposes
        event_dict = event.model_dump()
        event_dict['payload'] = updated_payload

        # Recreate the event (this preserves all other attributes)
        return event.__class__(**event_dict)

    def _create_backup(self) -> str:
        """Create backup of current event log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.event_store.data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        backup_path = backup_dir / f"events_backup_{timestamp}.log"

        if self.event_store.event_log_path.exists():
            shutil.copy2(self.event_store.event_log_path, backup_path)

        return str(backup_path)

    def _write_clean_events(self, clean_events: List[BaseEvent]) -> None:
        """Write clean events to a new event log."""
        # Remove current event log
        if self.event_store.event_log_path.exists():
            self.event_store.event_log_path.unlink()

        # Write clean events one by one
        for event in clean_events:
            self.event_store.append(event)

    def get_migration_preview(self) -> str:
        """Get a preview of what migration would do."""
        result = self.migrate_to_clean_log(dry_run=True)

        preview = [
            f"📊 Event Log Migration Preview",
            f"",
            f"📈 Events Analysis:",
            f"  • Total events: {result.events_processed}",
            f"  • Would keep: {result.events_kept}",
            f"  • Would filter out: {result.events_filtered}",
            f"",
        ]

        if result.projects_deleted:
            preview.append(f"🗑️  Projects to Delete ({len(result.projects_deleted)}):")
            for proj_id in result.projects_deleted[:5]:
                preview.append(f"  • {proj_id}")
            if len(result.projects_deleted) > 5:
                preview.append(f"  ... and {len(result.projects_deleted) - 5} more")
            preview.append("")

        if result.projects_renamed:
            preview.append(f"📝 Projects to Rename ({len(result.projects_renamed)}):")
            for old_id, new_id in list(result.projects_renamed.items())[:5]:
                preview.append(f"  • {old_id} → {new_id}")
            preview.append("")

        preview.append("💡 Use migration command with --execute to apply these changes")

        return "\n".join(preview)