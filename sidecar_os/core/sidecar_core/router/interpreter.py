"""Advanced pattern interpreter for contextual input analysis."""

import re
import asyncio
from datetime import datetime, UTC
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from uuid import uuid4

from ..events.schemas import BaseEvent, ProjectCreatedEvent, TaskCreatedEvent, TaskScheduledEvent, TaskDurationSetEvent, ClarificationRequestedEvent
from ..state.models import SidecarState, Project
from ..llm import LLMService, LLMConfig
from ..projects import ProjectMatcher, ProjectMatchResult


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
    # Updated to preserve full context instead of over-summarizing
    TASK_PATTERNS = [
        (r"(?:need to|should|must|have to|todo:?|task:?)\s+(.+)", 0.9),
        (r"((?:implement|build|create|add|write|develop|code|fix|debug|enhance|update|improve|optimize|refactor)\s+.+)", 0.8),
        (r"((?:call|contact|email|text|message|reach out to|respond to)\s+.+)", 0.85),
        (r"((?:review|check|look at|examine|investigate|analyze)\s+.+)", 0.75),
        (r"((?:finish|complete|wrap up|finalize)\s+.+)", 0.8),
        (r"((?:test|verify|validate|confirm|run)\s+.+)", 0.8),
        # Broader pattern to catch complete action phrases
        (r"([a-z]+(?:\s+[a-z]+)*\s+(?:and|&)\s+[a-z]+(?:\s+[a-z]+)*.*)", 0.7),  # "enhance model and complete backtesting"
    ]

    # Project patterns with confidence scores
    PROJECT_PATTERNS = [
        (r"^([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\s*:\s*", 0.95),  # Acronyms like "LPD:", "EB1:", "eb1a:", "lrp:"
        (r"(?:project|proj):?\s*([^:,\n]+)", 0.9),
        (r"(?:working on|focus on)\s+([^:,\n]+)", 0.8),
        (r"^([A-Za-z][a-zA-Z\s]*)\s*:\s*", 0.7),  # "Project Name: ..." - flexible case
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

    # Temporal patterns for due dates and durations
    DUE_DATE_PATTERNS = [
        (r"(?:by|due|deadline)\s+(today|tomorrow|this\s+week|next\s+week)", 0.9),
        (r"(?:by|due|deadline)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", 0.85),
        (r"(?:by|due|deadline)\s+(\d{1,2}\/\d{1,2}|\d{1,2}-\d{1,2})", 0.8),
        (r"(?:by|due|deadline)\s+(end\s+of\s+(?:week|month|year))", 0.75),
        (r"(?:by|due|deadline)\s+(in\s+\d+\s+(?:days?|weeks?))", 0.8),
        # Standalone temporal expressions (lower confidence)
        (r"\b(today)\b(?!\s+(?:is|was|will))", 0.7),  # "today" not followed by "is/was/will"
        (r"\b(tomorrow)\b(?!\s+(?:is|was|will))", 0.7),  # "tomorrow" not followed by "is/was/will"
        (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", 0.6),  # Standalone days
    ]

    DURATION_PATTERNS = [
        (r"\[(\d+(?:\.\d+)?)\s*(?:hrs?|hours?)\]", 0.95),  # [2 hrs], [1.5 hours]
        (r"\[(\d+)\s*(?:mins?|minutes?)\]", 0.95),         # [30min], [45 minutes]
        (r"\[(\d+(?:\.\d+)?)\s*h\]", 0.9),                # [1.5h]
        (r"\[(\d+)\s*m\]", 0.9),                          # [30m]
        (r"(?:takes?|duration|estimate)\s+(\d+)\s*(?:mins?|minutes?)", 0.8),
        (r"(?:takes?|duration|estimate)\s+(\d+(?:\.\d+)?)\s*(?:hrs?|hours?)", 0.8),
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

        # Initialize project matcher with same LLM service
        self.project_matcher = ProjectMatcher(llm_service=self.llm_service)

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

        # Extract clean text for task analysis (remove project prefix if found)
        clean_text = self._extract_clean_text_for_tasks(text, project_match)

        task_match = self._analyze_task_patterns(clean_text)
        promise_match = self._analyze_promise_patterns(clean_text)
        update_match = self._analyze_update_patterns(clean_text)

        # Analyze temporal patterns (due dates and durations) for tasks or project contexts
        temporal_result = None
        try:
            # Run temporal analysis if we have a good task match OR project match (which can infer tasks)
            should_analyze_temporal = (
                (task_match and task_match[1] > 0.6) or  # Explicit task match
                (project_match and project_match[1] > 0.7)  # Project match that could infer tasks
            )

            if should_analyze_temporal:
                temporal_result = asyncio.run(self._analyze_temporal_patterns(text))
            else:
                # Quick pattern-only temporal analysis to avoid LLM calls
                temporal_result = {
                    "due_date": None,
                    "duration_minutes": None,
                    "confidence": 0.0,
                    "method": "patterns",
                    "explanation": "No temporal analysis (low task/project confidence)"
                }
        except Exception as e:
            print(f"Temporal analysis failed: {e}")
            temporal_result = {
                "due_date": None,
                "duration_minutes": None,
                "confidence": 0.0,
                "method": "error",
                "explanation": f"Temporal analysis error: {e}"
            }

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

        # Handle task creation - but check for context and actionability
        if task_match:
            title, confidence, explanation = task_match

            # Check if task needs clarification despite high pattern confidence
            needs_task_clarification = self._task_needs_clarification(text, project_match, current_state)

            if not needs_task_clarification:
                # Task is clear and actionable - create it
                task_event = self._create_task_event(
                    title,
                    text,
                    project_id=project_match[0] if project_match and project_match[1] > 0.6 else None
                )
                events.append(task_event)
                task_id = task_event.payload["task_id"]

                # Add temporal events if temporal information was detected
                if temporal_result and temporal_result.get("confidence", 0) > 0.5:
                    if temporal_result.get("due_date") or temporal_result.get("due_date_text"):
                        scheduled_event = self._create_task_scheduled_event(
                            task_id,
                            temporal_result.get("due_date") or temporal_result.get("due_date_text")
                        )
                        if scheduled_event:
                            events.append(scheduled_event)

                    if temporal_result.get("duration_minutes"):
                        duration_event = self._create_task_duration_event(
                            task_id,
                            temporal_result["duration_minutes"]
                        )
                        events.append(duration_event)

                # Update explanations
                task_explanation = f"Created task '{title}' ({confidence:.0%} confidence)"
                if temporal_result and temporal_result.get("confidence", 0) > 0.5:
                    task_explanation += f" with {temporal_result['method']} temporal parsing"
                explanations.append(task_explanation)
                max_confidence = max(max_confidence, confidence)
            else:
                # Task pattern matched but needs clarification for context
                # Don't override the pattern confidence, but flag for clarification
                max_confidence = max(max_confidence, confidence)  # Keep actual pattern confidence
                explanations.append(f"Task needs clarification despite {confidence:.0%} pattern match")

                # Generate clarification questions directly
                clarification_questions = self._generate_clarification_questions(text, current_state)
                if clarification_questions:
                    clarification_event = self._create_clarification_event(text, clarification_questions)
                    events.append(clarification_event)
        elif project_match and project_match[1] > 0.7:
            remaining_text = self._extract_remaining_text_after_project(text)
            if remaining_text and len(remaining_text.split()) > 1:
                task_event = self._create_task_event(remaining_text, text, project_id=project_match[0])
                events.append(task_event)
                task_id = task_event.payload["task_id"]

                # Add temporal events if temporal information was detected for project-inferred tasks
                if temporal_result and temporal_result.get("confidence", 0) > 0.5:
                    if temporal_result.get("due_date") or temporal_result.get("due_date_text"):
                        scheduled_event = self._create_task_scheduled_event(
                            task_id,
                            temporal_result.get("due_date") or temporal_result.get("due_date_text")
                        )
                        if scheduled_event:
                            events.append(scheduled_event)

                    if temporal_result.get("duration_minutes"):
                        duration_event = self._create_task_duration_event(
                            task_id,
                            temporal_result["duration_minutes"]
                        )
                        events.append(duration_event)

                explanations.append(f"Inferred task from project context ({project_match[1]:.0%} confidence)")
                if temporal_result and temporal_result.get("confidence", 0) > 0.5:
                    explanations.append(f"with {temporal_result['method']} temporal parsing")
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

    def _extract_clean_text_for_tasks(self, text: str, project_match: Optional[tuple]) -> str:
        """Extract clean text for task analysis by removing project prefix.

        Args:
            text: Original input text
            project_match: Result of project pattern matching

        Returns:
            Text with project prefix removed for better task extraction
        """
        if not project_match:
            return text

        # Try to remove project prefix patterns
        for pattern, _ in self.PROJECT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Remove the matched project prefix
                clean_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                if clean_text:  # Make sure we didn't remove everything
                    return clean_text

        # Fallback: return original text if we couldn't clean it
        return text

    def _get_recent_context(self, state: SidecarState, limit: int = 10) -> List[BaseEvent]:
        """Get recent events for context analysis."""
        # This would normally come from event store, but for now return empty list
        # In full implementation, this would query the last N events
        return []

    def _analyze_project_patterns(self, text: str, state: SidecarState, recent_events: List[BaseEvent]) -> Optional[tuple]:
        """Analyze text for project patterns using semantic matching.

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

                # Use ProjectMatcher for semantic analysis
                try:
                    existing_projects = {
                        proj.project_id: proj.name
                        for proj in state.projects.values()
                    }

                    # Run semantic matching
                    match_result = asyncio.run(
                        self.project_matcher.find_best_match(project_name, existing_projects)
                    )

                    if match_result.matched_project_id:
                        # Found semantic match with existing project
                        confidence = min(base_confidence + 0.2, 1.0)  # Boost for semantic match
                        if confidence > best_confidence:
                            best_match = (
                                match_result.matched_project_id,
                                confidence,
                                f"Semantic match: '{project_name}' → '{match_result.matched_project_id}' ({match_result.confidence_score:.0%})"
                            )
                            best_confidence = confidence
                    else:
                        # No match, create new project with normalized name
                        if self._is_valid_project_name(project_name):
                            confidence = base_confidence
                            if confidence > best_confidence:
                                best_match = (
                                    match_result.canonical_id,
                                    confidence,
                                    f"New project: '{project_name}' → '{match_result.canonical_id}'"
                                )
                                best_confidence = confidence

                except Exception as e:
                    print(f"Project matching failed, falling back to basic logic: {e}")
                    # Fallback to original logic
                    existing_project = state.find_project_by_alias(project_name)
                    if existing_project:
                        confidence = min(base_confidence + 0.1, 1.0)
                        if confidence > best_confidence:
                            best_match = (existing_project.project_id, confidence, f"Matched existing project alias: {project_name}")
                            best_confidence = confidence
                    elif self._is_valid_project_name(project_name):
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

    def _analyze_due_date_patterns(self, text: str) -> Optional[tuple]:
        """Analyze text for due date patterns.

        Returns:
            tuple: (due_date_text, confidence, explanation) or None
        """
        for pattern, confidence in self.DUE_DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                due_text = match.group(1).strip()
                return (due_text, confidence, f"Due date pattern: '{due_text}'")
        return None

    def _analyze_duration_patterns(self, text: str) -> Optional[tuple]:
        """Analyze text for duration patterns.

        Returns:
            tuple: (duration_minutes, confidence, explanation) or None
        """
        for pattern, confidence in self.DURATION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                duration_text = match.group(1).strip()
                try:
                    # Parse duration to minutes
                    if "hr" in pattern or "hour" in pattern or r"\s*h\]" in pattern:
                        duration_minutes = int(float(duration_text) * 60)
                    else:
                        duration_minutes = int(duration_text)

                    if duration_minutes > 0 and duration_minutes <= 24 * 60:  # Max 24 hours
                        return (duration_minutes, confidence, f"Duration: {duration_minutes}min")
                except (ValueError, TypeError):
                    continue
        return None

    async def _analyze_temporal_patterns(self, text: str) -> Dict[str, Any]:
        """Extract temporal information using hybrid pattern + LLM approach.

        Returns:
            Dictionary with temporal parsing results
        """
        # First try basic patterns (fast path)
        due_date_match = self._analyze_due_date_patterns(text)
        duration_match = self._analyze_duration_patterns(text)

        # If both patterns found with high confidence, use them
        if (due_date_match and due_date_match[1] > 0.8 and
            duration_match and duration_match[1] > 0.8):
            return {
                "due_date_text": due_date_match[0],
                "duration_minutes": duration_match[0],
                "confidence": min(due_date_match[1], duration_match[1]),
                "method": "patterns",
                "explanation": f"{due_date_match[2]}, {duration_match[2]}"
            }

        # Fall back to LLM for complex temporal expressions
        if self.config.use_llm and self.llm_service:
            try:
                llm_result = await self.llm_service.parse_temporal_expressions(text)

                # Combine pattern results with LLM results
                final_result = {
                    "due_date": llm_result.get("due_date"),
                    "duration_minutes": llm_result.get("duration_minutes"),
                    "confidence": llm_result.get("confidence", 0.5),
                    "method": "llm",
                    "explanation": llm_result.get("explanation", "LLM temporal parsing")
                }

                # Override with high-confidence pattern results if available
                if due_date_match and due_date_match[1] > 0.8:
                    final_result["due_date_text"] = due_date_match[0]
                    final_result["confidence"] = max(final_result["confidence"], due_date_match[1])
                    final_result["method"] = "hybrid"

                if duration_match and duration_match[1] > 0.8:
                    final_result["duration_minutes"] = duration_match[0]
                    final_result["confidence"] = max(final_result["confidence"], duration_match[1])
                    final_result["method"] = "hybrid"

                return final_result

            except Exception as e:
                print(f"LLM temporal parsing failed: {e}")

        # Return pattern results if available, even if low confidence
        result = {
            "due_date": None,
            "duration_minutes": None,
            "confidence": 0.0,
            "method": "patterns",
            "explanation": "No temporal patterns detected"
        }

        if due_date_match:
            result["due_date_text"] = due_date_match[0]
            result["confidence"] = due_date_match[1]
            result["explanation"] = due_date_match[2]

        if duration_match:
            result["duration_minutes"] = duration_match[0]
            result["confidence"] = max(result["confidence"], duration_match[1])
            if "explanation" in result and result["explanation"] != "No temporal patterns detected":
                result["explanation"] += f", {duration_match[2]}"
            else:
                result["explanation"] = duration_match[2]

        return result

    def _task_needs_clarification(self, text: str, project_match: Optional[tuple], current_state: SidecarState) -> bool:
        """Check if a task pattern match needs clarification for context/actionability."""

        # Check for vague reference patterns using regex for flexibility
        # Exclude common development terms that are actually specific
        vague_patterns = [
            r"\bthe\s+team\b", r"\bthe\s+meeting\b", r"\bthe\s+call\b",
            r"\bthe\s+documents?\b", r"\bthe\s+files?\b", r"\bthe\s+thing\b",
            r"\bthe\s+issue\b", r"\bthe\s+problem\b", r"\bthe\s+project\b",
            r"\bthat\s+\w*\s*meeting\b", r"\bthat\s+\w*\s*call\b",
            r"\bthat\s+\w*\s*document\b", r"\bthat\s+\w*\s*thing\b",
            r"\bthis\s+\w*\s*meeting\b", r"\bthis\s+\w*\s*call\b",
            r"\bthe\s+requirements?\b", r"\bthe\s+specs?\b"
        ]

        # Exclude well-known development terms that don't need clarification
        specific_terms = ["pr", "pull request", "bug", "feature", "api", "database", "db", "server",
                         "client", "user", "admin", "login", "authentication", "auth", "code", "repo",
                         "repository", "branch", "merge", "deploy", "deployment", "test", "tests"]

        text_lower = text.lower()

        # Check for vague patterns but exclude specific development terms
        has_vague_references = False
        for pattern in vague_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract the word after "the"/"that"/"this"
                if "the " in match.group():
                    word_after = match.group().split()[-1]
                else:
                    word_after = match.group().split()[-1] if match.groups() else ""

                # If it's not a specific known term, consider it vague
                if word_after.lower() not in specific_terms:
                    has_vague_references = True
                    break
            if has_vague_references:
                break

        # Check if task lacks project context
        has_no_project_context = not project_match or project_match[1] < 0.6

        # Check for ambiguous pronouns
        ambiguous_pronouns = ["they", "them", "it", "this", "that", "these", "those"]
        has_ambiguous_pronouns = any(f" {pronoun} " in f" {text_lower} " for pronoun in ambiguous_pronouns)

        # Check if task contains question words suggesting uncertainty
        question_indicators = ["which", "what", "who", "when", "where", "how"]
        has_question_indicators = any(word in text_lower for word in question_indicators)

        # Debug for testing
        # print(f"CLARIFICATION DEBUG: text='{text}', vague={has_vague_references}, no_proj={has_no_project_context}, "
        #       f"pronouns={has_ambiguous_pronouns}, questions={has_question_indicators}")

        # Task needs clarification if:
        # 1. Has vague references AND no clear project context
        # 2. Has ambiguous pronouns without project context
        # 3. Contains question words suggesting uncertainty
        if has_vague_references and has_no_project_context:
            return True

        if has_ambiguous_pronouns and has_no_project_context:
            return True

        if has_question_indicators:
            return True

        # Additional check: if it's a very generic task without specific details
        # Check for generic patterns only when they're standalone or very vague
        generic_patterns = [
            r"\bfollow up\b(?:\s+on)?\s*$",  # "follow up" at end or "follow up on"
            r"\bcheck on\b\s*$",            # "check on" at end
            r"\blook into\b\s*$",           # "look into" at end
            r"\bwork on\b\s*$",             # "work on" at end
            r"\bhandle\b\s*$",              # "handle" at end
            r"\bdeal with\b\s*$",           # "deal with" at end
            r"\btake care of\b\s*$",        # "take care of" at end
            r"\breview\b\s*$",              # "review" alone at end
            r"\bupdate\b\s*$",              # "update" alone at end
            r"\bfix\b\s*$"                  # "fix" alone at end
        ]

        is_generic_task = any(re.search(pattern, text_lower) for pattern in generic_patterns)
        is_very_short = len(text.split()) <= 4  # Make this more restrictive

        if is_generic_task and is_very_short and has_no_project_context:
            return True

        return False

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

        # Mixed case: acronym + title case (e.g., "RSR Cube", "API Gateway", "DB Migration")
        if re.match(r'^[A-Z0-9]{2,10}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', name) and len(name) <= 50:
            return True

        # Mixed case: title + acronym (e.g., "Project API", "Service DB")
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[A-Z0-9]{2,10}$', name) and len(name) <= 50:
            return True

        return False

    def _generate_project_id(self, name: str) -> str:
        """Generate a project ID from a project name."""
        # Use name as ID for simplicity, could be more sophisticated
        return name.lower().replace(' ', '-')

    def _create_project_event(self, project_id: str, original_text: str) -> ProjectCreatedEvent:
        """Create a project created event with normalized naming."""
        # Use ProjectMatcher to get proper display name
        canonical_id, display_name = self.project_matcher.normalize_project_name(project_id)

        # Extract aliases from the original text
        aliases = []
        if ':' in original_text:
            prefix = original_text.split(':', 1)[0].strip()
            if prefix != canonical_id:
                aliases.append(prefix)

        return ProjectCreatedEvent(
            payload={
                "project_id": canonical_id,
                "name": display_name,
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

    def _create_task_scheduled_event(self, task_id: str, due_date_info: Any) -> Optional[TaskScheduledEvent]:
        """Create a task scheduled event with due date information."""
        try:
            # Handle different due date formats
            if isinstance(due_date_info, str):
                # Check if it's already an ISO format date string (with or without timezone)
                if 'T' in due_date_info and (due_date_info.endswith('Z') or '+' in due_date_info or '-' in due_date_info[-6:] or len(due_date_info) >= 19):
                    # Already in ISO format (2026-02-11T23:59:00 or 2026-02-11T23:59:00+00:00)
                    scheduled_for = due_date_info
                    # Add timezone if missing
                    if 'T' in scheduled_for and not (scheduled_for.endswith('Z') or '+' in scheduled_for or scheduled_for.endswith('+00:00')):
                        scheduled_for += '+00:00'  # Add UTC timezone
                else:
                    # It's a relative date text like "Friday", "tomorrow" - convert using simple logic
                    scheduled_for = self._convert_relative_date_to_iso(due_date_info)
            else:
                # Assume it's already in proper format
                scheduled_for = str(due_date_info)

            if scheduled_for:
                return TaskScheduledEvent(payload={
                    "task_id": task_id,
                    "scheduled_for": scheduled_for
                })

        except Exception as e:
            print(f"Failed to create scheduled event: {e}")

        return None

    def _create_task_duration_event(self, task_id: str, duration_minutes: int) -> TaskDurationSetEvent:
        """Create a task duration set event."""
        return TaskDurationSetEvent(payload={
            "task_id": task_id,
            "duration_minutes": duration_minutes
        })

    def _convert_relative_date_to_iso(self, relative_date: str) -> Optional[str]:
        """Convert relative date strings to ISO format dates."""
        from datetime import datetime, timedelta

        now = datetime.now(UTC)
        relative_lower = relative_date.lower().strip()

        # Handle common relative dates
        if relative_lower in ["today"]:
            target_date = now
        elif relative_lower in ["tomorrow"]:
            target_date = now + timedelta(days=1)
        elif relative_lower in ["this week"]:
            # End of this week (Sunday)
            days_until_sunday = 6 - now.weekday()
            target_date = now + timedelta(days=days_until_sunday)
        elif relative_lower in ["next week"]:
            # End of next week
            days_until_next_sunday = 6 - now.weekday() + 7
            target_date = now + timedelta(days=days_until_next_sunday)
        elif relative_lower in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            # Next occurrence of the specified weekday
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            target_weekday = weekdays.index(relative_lower)
            days_ahead = target_weekday - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            target_date = now + timedelta(days=days_ahead)
        else:
            # Can't parse this relative date
            return None

        # Set time to end of day (11:59 PM) for due dates
        target_date = target_date.replace(hour=23, minute=59, second=59, microsecond=0)
        return target_date.isoformat()

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