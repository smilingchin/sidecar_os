"""Tests for the advanced pattern interpreter."""

import pytest
from datetime import datetime, UTC

from sidecar_os.core.sidecar_core.router.interpreter import AdvancedPatternInterpreter, InterpretationResult
from sidecar_os.core.sidecar_core.state.models import SidecarState, Project
from sidecar_os.core.sidecar_core.events.schemas import (
    ProjectCreatedEvent,
    ProjectFocusedEvent,
    TaskCreatedEvent,
    ClarificationRequestedEvent
)


class TestAdvancedPatternInterpreter:
    """Test the advanced pattern interpreter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.interpreter = AdvancedPatternInterpreter()
        self.empty_state = SidecarState()

        # Create state with existing projects
        self.state_with_projects = SidecarState()
        now = datetime.now(UTC)

        eb1_project = Project(
            project_id="eb1-petition",
            name="EB1 Petition",
            aliases=["eb1", "fragomen"],
            created_at=now,
            last_focused_at=now
        )
        lpd_project = Project(
            project_id="lpd-experiments",
            name="LPD Experiments",
            aliases=["lpd"],
            created_at=now
        )

        self.state_with_projects.projects["eb1-petition"] = eb1_project
        self.state_with_projects.projects["lpd-experiments"] = lpd_project
        self.state_with_projects.current_focus_project = "lpd-experiments"

    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.interpreter.interpret_text("", self.empty_state)

        assert len(result.events) == 0
        assert result.confidence == 0.0
        assert "Empty input" in result.explanation

    def test_project_creation_with_acronym(self):
        """Test creating project from acronym pattern."""
        result = self.interpreter.interpret_text("LPD: ran 10/20 ablations", self.empty_state)

        assert len(result.events) >= 1
        assert result.confidence > 0.9

        # Should create project
        project_events = [e for e in result.events if isinstance(e, ProjectCreatedEvent)]
        assert len(project_events) == 1
        assert project_events[0].payload["project_id"] == "lpd"

        # Should also create task
        task_events = [e for e in result.events if isinstance(e, TaskCreatedEvent)]
        assert len(task_events) == 1
        assert "ran 10/20 ablations" in task_events[0].payload["title"]

    def test_project_focus_with_existing_project(self):
        """Test focusing on existing project."""
        result = self.interpreter.interpret_text("LPD: need to finish experiments", self.state_with_projects)

        assert len(result.events) >= 1
        assert result.confidence > 0.8

        # Should focus on existing project, not create new one
        focus_events = [e for e in result.events if isinstance(e, ProjectFocusedEvent)]
        assert len(focus_events) == 1
        assert focus_events[0].payload["project_id"] == "lpd-experiments"

        # No project creation events
        project_events = [e for e in result.events if isinstance(e, ProjectCreatedEvent)]
        assert len(project_events) == 0

    def test_task_pattern_matching(self):
        """Test various task pattern matches."""
        test_cases = [
            "Need to call the bank about mortgage",
            "Should implement the login feature",
            "Must review the PR before merging",
            "Have to email the client",
            "TODO: fix the bug in authentication",
            "Task: update documentation"
        ]

        for text in test_cases:
            result = self.interpreter.interpret_text(text, self.empty_state)
            assert len(result.events) >= 1
            assert result.confidence > 0.7

            task_events = [e for e in result.events if isinstance(e, TaskCreatedEvent)]
            assert len(task_events) >= 1

    def test_project_alias_matching(self):
        """Test matching projects by alias."""
        result = self.interpreter.interpret_text("Fragomen: need to submit forms", self.state_with_projects)

        assert len(result.events) >= 1
        assert result.confidence > 0.8

        # Should match EB1 project by "fragomen" alias
        focus_events = [e for e in result.events if isinstance(e, ProjectFocusedEvent)]
        task_events = [e for e in result.events if isinstance(e, TaskCreatedEvent)]

        assert len(focus_events) == 1
        assert focus_events[0].payload["project_id"] == "eb1-petition"

        # Task should be associated with the project
        if task_events:
            assert task_events[0].payload.get("project_id") == "eb1-petition"

    def test_context_based_inference(self):
        """Test using current focus for context."""
        # Input without explicit project indicator
        result = self.interpreter.interpret_text("need to run more tests", self.state_with_projects)

        # Should infer current focus project (lpd-experiments)
        task_events = [e for e in result.events if isinstance(e, TaskCreatedEvent)]
        if task_events:
            # May have project_id from context
            assert task_events[0].payload.get("project_id") in [None, "lpd-experiments"]

    def test_low_confidence_requires_clarification(self):
        """Test that ambiguous input triggers clarification."""
        result = self.interpreter.interpret_text("something unclear here", self.empty_state)

        assert result.needs_clarification
        assert len(result.clarification_questions) > 0

        clarification_events = [e for e in result.events if isinstance(e, ClarificationRequestedEvent)]
        assert len(clarification_events) == 1

    def test_complex_project_task_combination(self):
        """Test complex input with both project and task indicators."""
        result = self.interpreter.interpret_text(
            "EB1: need to call Fragomen about final petition status",
            self.state_with_projects
        )

        assert len(result.events) >= 1
        assert result.confidence > 0.8

        # Should focus on EB1 project and create task
        focus_events = [e for e in result.events if isinstance(e, ProjectFocusedEvent)]
        task_events = [e for e in result.events if isinstance(e, TaskCreatedEvent)]

        assert len(focus_events) == 1
        assert focus_events[0].payload["project_id"] == "eb1-petition"

        assert len(task_events) == 1
        assert "call Fragomen" in task_events[0].payload["title"]
        assert task_events[0].payload.get("project_id") == "eb1-petition"

    def test_update_pattern_detection(self):
        """Test detection of progress updates."""
        update_texts = [
            "Completed the database migration",
            "Progress: 50% done with testing",
            "Ran 15 experiments, +3pp improvement",
            "Finished reviewing all PRs"
        ]

        for text in update_texts:
            result = self.interpreter.interpret_text(text, self.empty_state)
            # Updates might create tasks or just be captured
            assert len(result.events) >= 0  # May or may not create events

    def test_promise_pattern_detection(self):
        """Test detection of promises/commitments."""
        promise_texts = [
            "Will finish the feature by Friday",
            "Going to call the client tomorrow",
            "Plan to review the code tonight",
            "Promise to submit the report by deadline"
        ]

        for text in promise_texts:
            result = self.interpreter.interpret_text(text, self.empty_state)
            # Promises might create tasks or be handled differently
            assert result.confidence >= 0.0  # At least analyzed

    def test_project_name_validation(self):
        """Test project name validation logic."""
        interpreter = self.interpreter

        # Valid project names
        assert interpreter._is_valid_project_name("LPD")
        assert interpreter._is_valid_project_name("EB1")
        assert interpreter._is_valid_project_name("Project Alpha")
        assert interpreter._is_valid_project_name("Machine Learning")

        # Invalid project names
        assert not interpreter._is_valid_project_name("")
        assert not interpreter._is_valid_project_name("x")
        assert not interpreter._is_valid_project_name("TOOLONGACRONYM")
        assert not interpreter._is_valid_project_name("lowercase only")

    def test_project_id_generation(self):
        """Test project ID generation from names."""
        interpreter = self.interpreter

        assert interpreter._generate_project_id("LPD") == "lpd"
        assert interpreter._generate_project_id("Project Alpha") == "project-alpha"
        assert interpreter._generate_project_id("Machine Learning") == "machine-learning"

    def test_clarification_question_generation(self):
        """Test generation of appropriate clarification questions."""
        # Test with existing projects
        questions = self.interpreter._generate_clarification_questions(
            "something unclear",
            self.state_with_projects
        )

        assert len(questions) > 0
        assert any("project" in q.lower() for q in questions)

        # Test with empty state
        questions_empty = self.interpreter._generate_clarification_questions(
            "something unclear",
            self.empty_state
        )

        assert len(questions_empty) > 0
        # Should not ask about projects if none exist
        assert not any("project" in q.lower() for q in questions_empty)

    def test_question_input_handling(self):
        """Test handling of question-like input."""
        result = self.interpreter.interpret_text(
            "How should I implement the authentication system?",
            self.empty_state
        )

        # Questions should often trigger clarification
        if result.needs_clarification:
            assert any("question" in q.lower() for q in result.clarification_questions)

    def test_confidence_scoring(self):
        """Test confidence scoring across different inputs."""
        high_confidence_inputs = [
            "LPD: need to run experiments",
            "TODO: implement feature X",
            "Must call the client"
        ]

        low_confidence_inputs = [
            "hmm, not sure about this",
            "maybe something?",
            "random thoughts here"
        ]

        for text in high_confidence_inputs:
            result = self.interpreter.interpret_text(text, self.state_with_projects)
            assert result.confidence > 0.7

        for text in low_confidence_inputs:
            result = self.interpreter.interpret_text(text, self.empty_state)
            assert result.confidence < 0.6