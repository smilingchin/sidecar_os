"""Tests for project-related functionality."""

import pytest
from datetime import datetime, UTC

from sidecar_os.core.sidecar_core.events.schemas import (
    ProjectCreatedEvent,
    ProjectFocusedEvent,
    ClarificationRequestedEvent
)
from sidecar_os.core.sidecar_core.state.models import Project, ClarificationRequest, SidecarState
from sidecar_os.core.sidecar_core.state.projector import StateProjector


class TestProjectEvents:
    """Test project-related event schemas."""

    def test_project_created_event(self):
        """Test ProjectCreatedEvent schema validation."""
        event = ProjectCreatedEvent(
            event_type="project_created",
            payload={
                "project_id": "proj-1",
                "name": "Test Project",
                "description": "A test project",
                "aliases": ["test", "proj1"]
            }
        )

        assert event.event_type == "project_created"
        assert event.payload["name"] == "Test Project"
        assert "test" in event.payload["aliases"]

    def test_project_focused_event(self):
        """Test ProjectFocusedEvent schema validation."""
        event = ProjectFocusedEvent(
            event_type="project_focused",
            payload={
                "project_id": "proj-1",
                "context": "Starting work session"
            }
        )

        assert event.event_type == "project_focused"
        assert event.payload["project_id"] == "proj-1"

    def test_clarification_requested_event(self):
        """Test ClarificationRequestedEvent schema validation."""
        event = ClarificationRequestedEvent(
            event_type="clarification_requested",
            payload={
                "source_event_id": "event-123",
                "questions": ["Is this a task or a note?", "Which project does this belong to?"]
            }
        )

        assert event.event_type == "clarification_requested"
        assert len(event.payload["questions"]) == 2


class TestProjectStateModels:
    """Test project-related state models."""

    def test_project_creation(self):
        """Test Project model creation."""
        now = datetime.now(UTC)

        project = Project(
            project_id="proj-1",
            name="Test Project",
            description="A test project",
            aliases=["test", "proj1"],
            created_at=now
        )

        assert project.project_id == "proj-1"
        assert project.name == "Test Project"
        assert project.focus_count == 0
        assert project.last_focused_at is None

    def test_clarification_request_creation(self):
        """Test ClarificationRequest model creation."""
        now = datetime.now(UTC)

        clarification = ClarificationRequest(
            request_id="req-1",
            source_event_id="event-123",
            questions=["What type is this?"],
            created_at=now
        )

        assert clarification.request_id == "req-1"
        assert not clarification.resolved

    def test_sidecar_state_project_methods(self):
        """Test SidecarState project-related methods."""
        state = SidecarState()
        now = datetime.now(UTC)

        # Add test project
        project = Project(
            project_id="proj-1",
            name="Test Project",
            aliases=["test", "proj1"],
            created_at=now,
            last_focused_at=now
        )
        state.projects["proj-1"] = project

        # Test find_project_by_alias
        found = state.find_project_by_alias("test")
        assert found is not None
        assert found.project_id == "proj-1"

        found_by_name = state.find_project_by_alias("Test Project")
        assert found_by_name is not None
        assert found_by_name.project_id == "proj-1"

        # Test case insensitive search
        found_case = state.find_project_by_alias("TEST")
        assert found_case is not None

        # Test not found
        not_found = state.find_project_by_alias("nonexistent")
        assert not_found is None

        # Test get_recent_projects
        recent = state.get_recent_projects()
        assert len(recent) == 1
        assert recent[0].project_id == "proj-1"


class TestProjectStateProjection:
    """Test project event projection to state."""

    def setup_method(self):
        """Set up test fixtures."""
        self.projector = StateProjector()

    def test_project_created_projection(self):
        """Test projection of project_created event."""
        event = ProjectCreatedEvent(
            payload={
                "project_id": "proj-1",
                "name": "Test Project",
                "description": "A test project",
                "aliases": ["test", "proj1"]
            }
        )

        state = self.projector.project_state([event])

        assert len(state.projects) == 1
        project = state.projects["proj-1"]
        assert project.name == "Test Project"
        assert project.description == "A test project"
        assert "test" in project.aliases
        assert project.focus_count == 0

    def test_project_focused_projection(self):
        """Test projection of project_focused event."""
        created_event = ProjectCreatedEvent(
            payload={
                "project_id": "proj-1",
                "name": "Test Project"
            }
        )

        focused_event = ProjectFocusedEvent(
            payload={
                "project_id": "proj-1"
            }
        )

        state = self.projector.project_state([created_event, focused_event])

        assert len(state.projects) == 1
        project = state.projects["proj-1"]
        assert project.focus_count == 1
        assert project.last_focused_at == focused_event.timestamp
        assert state.current_focus_project == "proj-1"

    def test_clarification_requested_projection(self):
        """Test projection of clarification_requested event."""
        event = ClarificationRequestedEvent(
            payload={
                "source_event_id": "event-123",
                "questions": ["What type is this?"]
            }
        )

        state = self.projector.project_state([event])

        assert len(state.clarifications) == 1
        clarification = state.clarifications[event.event_id]
        assert clarification.source_event_id == "event-123"
        assert not clarification.resolved
        assert len(clarification.questions) == 1

    def test_stats_include_projects(self):
        """Test that stats include project counts."""
        project_event = ProjectCreatedEvent(
            payload={
                "project_id": "proj-1",
                "name": "Test Project"
            }
        )

        clarification_event = ClarificationRequestedEvent(
            payload={
                "source_event_id": "event-123",
                "questions": ["What type?"]
            }
        )

        state = self.projector.project_state([project_event, clarification_event])

        assert state.stats.project_count == 1
        assert state.stats.pending_clarifications == 1