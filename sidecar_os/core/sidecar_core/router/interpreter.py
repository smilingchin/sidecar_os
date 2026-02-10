"""Advanced pattern interpreter for contextual input analysis."""

import re
from datetime import datetime, UTC
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from uuid import uuid4

from ..events.schemas import BaseEvent, ProjectCreatedEvent, TaskCreatedEvent, ClarificationRequestedEvent
from ..state.models import SidecarState, Project


@dataclass
class InterpretationResult:
    """Result of input interpretation containing structured events."""

    events: List[BaseEvent]
    confidence: float  # 0.0 to 1.0
    explanation: str
    needs_clarification: bool = False
    clarification_questions: List[str] = None

    def __post_init__(self):
        if self.clarification_questions is None:
            self.clarification_questions = []


class AdvancedPatternInterpreter:
    """Advanced contextual pattern interpreter for structured input analysis."""

    # Task patterns with confidence scores
    TASK_PATTERNS = [
        (r"(?:need to|should|must|have to|todo:?|task:?)\s+(.+)", 0.9),
        (r"(?:implement|build|create|add|write|develop|code|fix|debug)\s+(.+)", 0.8),
        (r"(?:call|contact|email|text|message|reach out to)\s+(.+)", 0.85),
        (r"(?:review|check|look at|examine|investigate)\s+(.+)", 0.75),
        (r"(?:finish|complete|wrap up|finalize)\s+(.+)", 0.8),
        (r"(?:test|verify|validate|confirm)\s+(.+)", 0.8),
    ]

    # Project patterns with confidence scores
    PROJECT_PATTERNS = [
        (r"^([A-Z0-9]{2,5}):\s*", 0.95),  # Acronyms like "LPD:", "EB1:" with optional space after colon
        (r"(?:project|proj):?\s*([^:,\n]+)", 0.9),
        (r"(?:working on|focus on)\s+([^:,\n]+)", 0.8),
        (r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:\s+", 0.7),  # "Project Name: ..."
    ]

    # Promise/commitment patterns
    PROMISE_PATTERNS = [
        (r"(?:will|going to|plan to|intend to)\s+(.+)", 0.8),
        (r"(?:promise|commit|guarantee)\s+(?:to\s+)?(.+)", 0.9),
        (r"(?:by|due|deadline)\s+(today|tomorrow|this week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)", 0.85),
    ]

    # Update/progress patterns
    UPDATE_PATTERNS = [
        (r"(?:completed|finished|done with|wrapped up)\s+(.+)", 0.9),
        (r"(?:progress|status|update):?\s+(.+)", 0.8),
        (r"(?:ran|executed|performed|did)\s+(.+)", 0.75),
        (r"(?:\+\d+pp|\+\d+\%|\-\d+pp|\-\d+\%|improved|worse|better)", 0.8),  # Performance metrics
    ]

    def __init__(self):
        """Initialize the interpreter."""
        self._project_cache: Dict[str, str] = {}  # alias -> project_id mapping

    def interpret_text(self, text: str, current_state: SidecarState) -> InterpretationResult:
        """Interpret input text and generate structured events.

        Args:
            text: Raw input text to interpret
            current_state: Current system state for context

        Returns:
            InterpretationResult with events and confidence
        """
        text = text.strip()
        if not text:
            return InterpretationResult(
                events=[],
                confidence=0.0,
                explanation="Empty input"
            )

        # Update project cache from current state
        self._update_project_cache(current_state)

        # Get recent context for interpretation
        recent_events = self._get_recent_context(current_state)

        # Analyze patterns
        project_match = self._analyze_project_patterns(text, current_state, recent_events)
        task_match = self._analyze_task_patterns(text)
        promise_match = self._analyze_promise_patterns(text)
        update_match = self._analyze_update_patterns(text)

        # Build interpretation result
        events = []
        explanations = []
        max_confidence = 0.0

        # Handle project inference/creation
        if project_match:
            project_id, confidence, explanation = project_match
            if confidence > 0.7:
                if project_id not in current_state.projects:
                    # Create new project
                    events.append(self._create_project_event(project_id, text))
                    explanations.append(f"Created project '{project_id}' ({confidence:.0%} confidence)")
                else:
                    # Focus existing project
                    events.append(self._create_focus_event(project_id))
                    explanations.append(f"Focused on project '{project_id}' ({confidence:.0%} confidence)")
                max_confidence = max(max_confidence, confidence)

        # Handle task creation
        if task_match:
            title, confidence, explanation = task_match
            task_event = self._create_task_event(
                title,
                text,
                project_id=project_match[0] if project_match and project_match[1] > 0.6 else None
            )
            events.append(task_event)
            explanations.append(f"Created task '{title}' ({confidence:.0%} confidence)")
            max_confidence = max(max_confidence, confidence)
        elif project_match and project_match[1] > 0.7:
            # If we have a clear project match but no task pattern,
            # check if the remaining text after project indicator could be a task
            remaining_text = self._extract_remaining_text_after_project(text)
            if remaining_text and len(remaining_text.split()) > 1:
                task_event = self._create_task_event(
                    remaining_text,
                    text,
                    project_id=project_match[0]
                )
                events.append(task_event)
                explanations.append(f"Inferred task from project context ({project_match[1]:.0%} confidence)")
                max_confidence = max(max_confidence, project_match[1])

        # Check if we need clarification
        needs_clarification = max_confidence < 0.6 and len(events) == 0
        clarification_questions = []

        if needs_clarification:
            clarification_questions = self._generate_clarification_questions(text, current_state)
            if clarification_questions:
                events.append(self._create_clarification_event(text, clarification_questions))

        # Generate explanation
        if explanations:
            explanation = "; ".join(explanations)
        elif needs_clarification:
            explanation = f"Low confidence interpretation ({max_confidence:.0%}), requesting clarification"
        else:
            explanation = f"No clear patterns detected in: '{text}'"

        return InterpretationResult(
            events=events,
            confidence=max_confidence,
            explanation=explanation,
            needs_clarification=needs_clarification,
            clarification_questions=clarification_questions
        )

    def _update_project_cache(self, state: SidecarState) -> None:
        """Update project alias cache from current state."""
        self._project_cache.clear()
        for project in state.projects.values():
            # Map name to project_id
            self._project_cache[project.name.lower()] = project.project_id
            # Map aliases to project_id
            for alias in project.aliases:
                self._project_cache[alias.lower()] = project.project_id

    def _get_recent_context(self, state: SidecarState, limit: int = 10) -> List[BaseEvent]:
        """Get recent events for context analysis."""
        # This would normally come from event store, but for now return empty list
        # In full implementation, this would query the last N events
        return []

    def _analyze_project_patterns(self, text: str, state: SidecarState, recent_events: List[BaseEvent]) -> Optional[tuple]:
        """Analyze text for project patterns and context.

        Returns:
            tuple: (project_id, confidence, explanation) or None
        """
        best_match = None
        best_confidence = 0.0

        # Check explicit project patterns
        for pattern, base_confidence in self.PROJECT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_name = match.group(1).strip()

                # Boost confidence if project already exists
                existing_project = state.find_project_by_alias(project_name)
                if existing_project:
                    confidence = min(base_confidence + 0.1, 1.0)
                    if confidence > best_confidence:
                        best_match = (existing_project.project_id, confidence, f"Matched existing project alias: {project_name}")
                        best_confidence = confidence
                else:
                    # Check if this looks like a valid project name
                    if self._is_valid_project_name(project_name):
                        confidence = base_confidence
                        if confidence > best_confidence:
                            best_match = (self._generate_project_id(project_name), confidence, f"New project detected: {project_name}")
                            best_confidence = confidence

        # Check context-based inference
        if state.current_focus_project and best_confidence < 0.8:
            context_confidence = 0.6  # Lower confidence for context-based
            if context_confidence > best_confidence:
                best_match = (state.current_focus_project, context_confidence, "Inferred from current focus")

        return best_match

    def _analyze_task_patterns(self, text: str) -> Optional[tuple]:
        """Analyze text for task patterns.

        Returns:
            tuple: (title, confidence, explanation) or None
        """
        for pattern, confidence in self.TASK_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                return (title, confidence, f"Task pattern match: {pattern}")

        # Fallback: if text looks task-like, treat as low-confidence task
        if len(text.split()) > 2 and any(word in text.lower() for word in ['need', 'should', 'must', 'todo', 'task']):
            return (text, 0.5, "Weak task indicators")

        return None

    def _analyze_promise_patterns(self, text: str) -> Optional[tuple]:
        """Analyze text for promise/commitment patterns."""
        for pattern, confidence in self.PROMISE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                commitment = match.group(1).strip() if match.groups() else text
                return (commitment, confidence, f"Promise pattern match: {pattern}")
        return None

    def _analyze_update_patterns(self, text: str) -> Optional[tuple]:
        """Analyze text for progress update patterns."""
        for pattern, confidence in self.UPDATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                update = match.group(1).strip() if match.groups() else text
                return (update, confidence, f"Update pattern match: {pattern}")
        return None

    def _is_valid_project_name(self, name: str) -> bool:
        """Check if a string looks like a valid project name."""
        if not name or len(name) < 2:
            return False

        # Acronyms (2-5 uppercase letters or alphanumeric)
        if re.match(r'^[A-Z0-9]{2,5}$', name):
            return True

        # Title case names
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', name) and len(name) <= 50:
            return True

        return False

    def _generate_project_id(self, name: str) -> str:
        """Generate a project ID from a project name."""
        # Use name as ID for simplicity, could be more sophisticated
        return name.lower().replace(' ', '-')

    def _create_project_event(self, project_id: str, original_text: str) -> ProjectCreatedEvent:
        """Create a project created event."""
        # Extract aliases from the original text
        aliases = []
        if ':' in original_text:
            prefix = original_text.split(':', 1)[0].strip()
            if prefix != project_id:
                aliases.append(prefix)

        return ProjectCreatedEvent(
            payload={
                "project_id": project_id,
                "name": project_id.replace('-', ' ').title(),
                "aliases": aliases,
                "description": f"Auto-created from: {original_text}"
            }
        )

    def _create_focus_event(self, project_id: str) -> BaseEvent:
        """Create a project focused event."""
        from ..events.schemas import ProjectFocusedEvent
        return ProjectFocusedEvent(
            payload={
                "project_id": project_id
            }
        )

    def _create_task_event(self, title: str, original_text: str, project_id: Optional[str] = None) -> TaskCreatedEvent:
        """Create a task created event."""
        payload = {
            "task_id": str(uuid4()),
            "title": title,
            "description": f"Auto-created from: {original_text}",
            "created_from_event": ""  # Will be set by caller
        }

        if project_id:
            payload["project_id"] = project_id

        return TaskCreatedEvent(payload=payload)

    def _create_clarification_event(self, original_text: str, questions: List[str]) -> ClarificationRequestedEvent:
        """Create a clarification requested event."""
        return ClarificationRequestedEvent(
            payload={
                "source_event_id": "",  # Will be set by caller
                "questions": questions,
                "original_text": original_text
            }
        )

    def _generate_clarification_questions(self, text: str, state: SidecarState) -> List[str]:
        """Generate appropriate clarification questions."""
        questions = []

        # Check what's ambiguous - handle None values properly
        patterns = [
            self._analyze_project_patterns(text, state, []),
            self._analyze_task_patterns(text),
            self._analyze_promise_patterns(text),
            self._analyze_update_patterns(text)
        ]

        # Filter out None values and check if any patterns matched
        valid_patterns = [p for p in patterns if p is not None]
        if not valid_patterns:
            questions.append("Is this a task, note, or something else?")

        if len(state.projects) > 0:
            questions.append("Which project does this relate to?")

        if "?" in text or text.endswith("?"):
            questions.append("This looks like a question - should it be captured as a task to investigate?")

        return questions or ["What type of item is this and how should it be categorized?"]

    def _extract_remaining_text_after_project(self, text: str) -> str:
        """Extract text after project indicators."""
        # Try to find project pattern and extract what comes after it
        for pattern, _ in self.PROJECT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return text after the match
                end_pos = match.end()
                remaining = text[end_pos:].strip()
                if remaining.startswith(':'):
                    remaining = remaining[1:].strip()
                return remaining
        return text.strip()