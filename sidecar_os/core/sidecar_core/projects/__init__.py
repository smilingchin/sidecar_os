"""Project management module for Sidecar OS."""

from .matcher import ProjectMatcher, ProjectMatchResult
from .cleanup import ProjectCleanupManager, CleanupSuggestion
from .migration import EventLogMigrator, MigrationResult

__all__ = ['ProjectMatcher', 'ProjectMatchResult', 'ProjectCleanupManager', 'CleanupSuggestion', 'EventLogMigrator', 'MigrationResult']