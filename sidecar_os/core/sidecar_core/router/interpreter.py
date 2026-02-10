"""Advanced pattern interpreter for contextual input analysis."""

import re
import asyncio
from datetime import datetime, UTC
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from uuid import uuid4

from ..events.schemas import BaseEvent, ProjectCreatedEvent, TaskCreatedEvent, ClarificationRequestedEvent
from ..state.models import SidecarState, Project
from ..llm import LLMService, LLMConfig


@dataclass
class InterpretationResult:
    """Result of input interpretation containing structured events."""

    events: List[BaseEvent]
    confidence: float  # 0.0 to 1.0
    explanation: str
    needs_clarification: bool = False
    clarification_questions: List[str] = None
    used_llm: bool = False  # Whether LLM was invoked
    pattern_confidence: Optional[float] = None  # Original pattern confidence
    llm_confidence: Optional[float] = None  # LLM confidence if used
    analysis_method: str = "pattern"  # "pattern", "llm", or "hybrid"

    def __post_init__(self):
        if self.clarification_questions is None:
            self.clarification_questions = []


@dataclass
class InterpreterConfig:
    """Configuration for the hybrid interpreter."""
    use_llm: bool = True
    llm_confidence_threshold: float = 0.6  # Use LLM if pattern confidence < this
    immediate_clarification_threshold: float = 0.3  # Ask immediate questions if combined < this
    llm_provider: str = "bedrock"
    llm_model: str = "claude-opus-4.6"


class AdvancedPatternInterpreter:
    """Advanced contextual pattern interpreter for structured input analysis with LLM integration."""

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
        (r"^([A-Z0-9]+(?:\s+[A-Z0-9]+)*)\s*:\s*", 0.95),  # Acronyms like "LPD:", "EB1:", "UVP EU:"
        (r"(?:project|proj):?\s*([^:,\n]+)", 0.9),
        (r"(?:working on|focus on)\s+([^:,\n]+)", 0.8),
        (r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:\s*", 0.7),  # "Project Name: ..." with optional space after colon
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

    def __init__(self, config: Optional[InterpreterConfig] = None):
        """Initialize the interpreter with optional LLM integration."""
        self._project_cache: Dict[str, str] = {}  # alias -> project_id mapping
        self.config = config or InterpreterConfig()

        # Initialize LLM service if enabled
        self.llm_service: Optional[LLMService] = None
        if self.config.use_llm:
            try:
                llm_config = LLMConfig(
                    provider=self.config.llm_provider,
                    model=self.config.llm_model
                )
                self.llm_service = LLMService(config=llm_config)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM service: {e}")
                print("Falling back to pattern-only mode")
                self.config.use_llm = False

    def interpret_text(self, text: str, current_state: SidecarState) -> InterpretationResult:
        """Hybrid interpretation using patterns first, LLM fallback for ambiguous cases.

        Args:
            text: Raw input text to interpret
            current_state: Current system state for context

        Returns:
            InterpretationResult with events, confidence, and method used
        """
        text = text.strip()
        if not text:
            return InterpretationResult(
                events=[],
                confidence=0.0,
                explanation="Empty input",
                analysis_method="none"
            )

        # Step 1: Always try pattern analysis first (fast path)
        pattern_result = self._analyze_with_patterns(text, current_state)

        # Step 2: If pattern confidence is high enough, use it directly
        if pattern_result.confidence >= self.config.llm_confidence_threshold:
            pattern_result.analysis_method = "pattern"
            pattern_result.pattern_confidence = pattern_result.confidence
            return pattern_result

        # Step 3: Pattern confidence is low, try LLM if available
        llm_result = None
        if self.config.use_llm and self.llm_service:
            try:
                llm_result = asyncio.run(self._analyze_with_llm(text, current_state))
            except Exception as e:
                print(f"LLM analysis failed: {e}, falling back to patterns")

        # Step 4: Merge pattern and LLM results
        final_result = self._merge_interpretations(pattern_result, llm_result, text, current_state)

        return final_result

    def _analyze_with_patterns(self, text: str, current_state: SidecarState) -> InterpretationResult:
        """Analyze text using existing pattern matching logic."""
        # Update project cache from current state
        self._update_project_cache(current_state)

        # Get recent context for interpretation
        recent_events = self._get_recent_context(current_state)

        # Analyze patterns (existing logic)
        project_match = self._analyze_project_patterns(text, current_state, recent_events)
        task_match = self._analyze_task_patterns(text)
        promise_match = self._analyze_promise_patterns(text)
        update_match = self._analyze_update_patterns(text)

        # Build interpretation result (existing logic)
        events = []
        explanations = []
        max_confidence = 0.0

        # Handle project inference/creation
        if project_match:
            project_id, confidence, explanation = project_match
            if confidence > 0.7:
                if project_id not in current_state.projects:
                    events.append(self._create_project_event(project_id, text))
                    explanations.append(f"Created project '{project_id}' ({confidence:.0%} confidence)")
                else:
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
            remaining_text = self._extract_remaining_text_after_project(text)
            if remaining_text and len(remaining_text.split()) > 1:
                task_event = self._create_task_event(remaining_text, text, project_id=project_match[0])
                events.append(task_event)
                explanations.append(f"Inferred task from project context ({project_match[1]:.0%} confidence)")
                max_confidence = max(max_confidence, project_match[1])

        # Generate explanation
        if explanations:
            explanation = "; ".join(explanations)
        else:
            explanation = f"Pattern analysis: {max_confidence:.0%} confidence"

        return InterpretationResult(
            events=events,
            confidence=max_confidence,
            explanation=explanation,
            analysis_method="pattern"
        )

    async def _analyze_with_llm(self, text: str, current_state: SidecarState) -> Dict[str, Any]:
        """Analyze text using LLM for structured interpretation."""
        # Build context for LLM
        context = {
            "current_focus_project": current_state.current_focus_project,
            "recent_projects": [p.name for p in current_state.get_recent_projects(3)],
            "active_tasks_count": len(current_state.get_active_tasks()),
            "unprocessed_items": len(current_state.get_unprocessed_inbox())
        }

        # Use LLM service for interpretation
        llm_result = await self.llm_service.interpret_text(text, context=context)
        return llm_result

    def _merge_interpretations(
        self,
        pattern_result: InterpretationResult,
        llm_result: Optional[Dict[str, Any]],
        text: str,
        current_state: SidecarState
    ) -> InterpretationResult:
        """Merge pattern and LLM analysis results."""
        if not llm_result:
            # LLM failed, return pattern result
            pattern_result.analysis_method = "pattern_only"
            return self._handle_low_confidence(pattern_result, text, current_state)

        # Extract LLM confidence and data
        llm_confidence = llm_result.get("overall_confidence", 0.0)
        llm_projects = llm_result.get("projects", [])
        llm_tasks = llm_result.get("tasks", [])

        # Combine confidences (weighted average favoring higher values)
        combined_confidence = max(pattern_result.confidence, llm_confidence)

        # Build combined events
        events = list(pattern_result.events)  # Start with pattern events

        # Add LLM-identified projects if pattern didn't find them
        if llm_confidence > 0.7 and not any(isinstance(e, (ProjectCreatedEvent, type(self._create_focus_event("")))) for e in events):
            for llm_project in llm_projects:
                if llm_project.get("confidence", 0) > 0.7:
                    project_name = llm_project.get("name", "")
                    if project_name and not current_state.find_project_by_alias(project_name):
                        events.append(self._create_project_event(project_name.lower().replace(" ", "-"), text))

        # Add LLM-identified tasks if pattern didn't find them
        if llm_confidence > 0.7 and not any(isinstance(e, TaskCreatedEvent) for e in events):
            for llm_task in llm_tasks:
                if llm_task.get("confidence", 0) > 0.7:
                    task_title = llm_task.get("title", "")
                    if task_title:
                        events.append(self._create_task_event(task_title, text))

        # Determine if clarification is needed
        needs_clarification = combined_confidence < self.config.immediate_clarification_threshold
        clarification_questions = []

        if needs_clarification:
            # Generate intelligent clarification questions using both pattern and LLM insights
            clarification_questions = self._generate_intelligent_clarification(text, current_state, llm_result)
            if clarification_questions:
                events.append(self._create_clarification_event(text, clarification_questions))

        # Build explanation
        explanation_parts = []
        if pattern_result.confidence > 0.1:  # Lower threshold to show more info
            explanation_parts.append(f"Pattern: {pattern_result.confidence:.0%}")
        if llm_confidence > 0.1:  # Lower threshold to show more info
            explanation_parts.append(f"LLM: {llm_confidence:.0%}")

        # Always show something in parentheses
        if not explanation_parts:
            explanation_parts.append("Low confidence")

        explanation = f"Hybrid analysis ({', '.join(explanation_parts)}) - {combined_confidence:.0%} combined confidence"

        return InterpretationResult(
            events=events,
            confidence=combined_confidence,
            explanation=explanation,
            needs_clarification=needs_clarification,
            clarification_questions=clarification_questions,
            used_llm=True,
            pattern_confidence=pattern_result.confidence,
            llm_confidence=llm_confidence,
            analysis_method="hybrid"
        )

    def _handle_low_confidence(
        self,
        result: InterpretationResult,
        text: str,
        current_state: SidecarState
    ) -> InterpretationResult:
        """Handle low-confidence interpretations that need clarification."""
        if result.confidence < self.config.immediate_clarification_threshold:
            result.needs_clarification = True
            result.clarification_questions = self._generate_clarification_questions(text, current_state)
            if result.clarification_questions:
                result.events.append(self._create_clarification_event(text, result.clarification_questions))
            result.explanation = f"Low confidence ({result.confidence:.0%}), requesting clarification"

        return result

    def _generate_intelligent_clarification(
        self,
        text: str,
        current_state: SidecarState,
        llm_result: Dict[str, Any]
    ) -> List[str]:
        """Generate smart clarification questions based on LLM analysis."""
        questions = []

        # Use LLM insights to generate better questions
        llm_type = llm_result.get("type", "unknown")

        if llm_type == "mixed" or llm_type == "unknown":
            questions.append("Is this a task you need to do, meeting notes, or a status update?")

        if len(current_state.projects) > 0:
            projects_list = ", ".join([p.name for p in current_state.get_recent_projects(3)])
            questions.append(f"Which project is this related to? ({projects_list} or something else?)")

        if "task" in llm_type.lower() or any("task" in str(task) for task in llm_result.get("tasks", [])):
            questions.append("What specific action needs to be taken?")
            questions.append("Is there a deadline or priority for this?")

        # Fallback to basic questions if LLM didn't provide much insight
        if not questions:
            questions = self._generate_clarification_questions(text, current_state)

        return questions

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

        # Single acronyms (2-10 uppercase letters or alphanumeric)
        if re.match(r'^[A-Z0-9]{2,10}$', name):
            return True

        # Multi-word acronyms like "UVP EU", "API V2"
        if re.match(r'^[A-Z0-9]{2,10}(?:\s+[A-Z0-9]{1,10})+$', name):
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