"""Project management module for Sidecar OS."""

from .matcher import ProjectMatcher, ProjectMatchResult
from .cleanup import ProjectCleanupManager, CleanupSuggestion

__all__ = ['ProjectMatcher', 'ProjectMatchResult', 'ProjectCleanupManager', 'CleanupSuggestion']