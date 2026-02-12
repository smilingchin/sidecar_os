"""LLM service orchestrator and configuration management."""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, UTC

from .providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMMessage,
    LLMRole,
    LLMError,
)
from .providers.bedrock import BedrockProvider
from .providers.mock import MockProvider
from .usage_tracker import get_usage_tracker

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    provider: str = "bedrock"
    model: str = "claude-opus-4.6"
    max_tokens: int = 4000
    temperature: float = 0.1
    aws_profile: Optional[str] = None
    aws_region: str = "us-east-1"
    confidence_threshold: float = 0.8
    rate_limit: int = 60  # requests per minute
    fallback_to_patterns: bool = True
    cost_limit_daily: float = 10.0  # USD
    mock_delay: float = 0.1
    mock_fail_rate: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class LLMService:
    """Main LLM service that orchestrates providers and handles configuration."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM service.

        Args:
            config: LLM configuration, loads from file if None
        """
        self.config = config or self._load_config()
        self.provider: Optional[LLMProvider] = None
        self.request_count = 0
        self.daily_cost = 0.0

        # Initialize provider
        self._initialize_provider()

        logger.info(f"LLM service initialized with {self.config.provider} provider")

    def _load_config(self) -> LLMConfig:
        """Load configuration from environment and files."""
        # Start with defaults
        config = LLMConfig()

        # Load from environment variables
        config.provider = os.getenv("SIDECAR_LLM_PROVIDER", config.provider)
        config.model = os.getenv("SIDECAR_LLM_MODEL", config.model)
        config.aws_profile = os.getenv("SIDECAR_AWS_PROFILE", config.aws_profile)
        config.aws_region = os.getenv("SIDECAR_AWS_REGION", config.aws_region)

        # Try to load from config file
        config_path = Path.home() / ".config" / "sidecar" / "llm.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Update config with file values
                    for key, value in file_config.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                logger.info(f"Loaded LLM config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return config

    def _initialize_provider(self) -> None:
        """Initialize the LLM provider based on configuration."""
        try:
            if self.config.provider == "bedrock":
                self.provider = BedrockProvider(
                    region=self.config.aws_region,
                    aws_profile=self.config.aws_profile
                )
            elif self.config.provider == "mock":
                self.provider = MockProvider(
                    delay=self.config.mock_delay,
                    fail_rate=self.config.mock_fail_rate
                )
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

        except Exception as e:
            logger.error(f"Failed to initialize {self.config.provider} provider: {e}")
            # Fallback to mock provider
            logger.info("Falling back to mock provider")
            self.provider = MockProvider()
            self.config.provider = "mock"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            model: Model to use (uses config default if None)
            max_tokens: Max tokens to generate
            temperature: Generation temperature
            metadata: Additional metadata

        Returns:
            LLM response

        Raises:
            LLMError: Various LLM-related errors
        """
        # Check cost limits
        if self.daily_cost >= self.config.cost_limit_daily:
            raise LLMError(f"Daily cost limit reached: ${self.config.cost_limit_daily}")

        # Prepare request
        messages = [LLMMessage(role=LLMRole.USER, content=prompt)]

        request = LLMRequest(
            messages=messages,
            model=model or self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            system_prompt=system_prompt,
            metadata=metadata
        )

        # Estimate cost before making request
        cost_estimate = self.provider.estimate_cost(request)
        estimated_cost = cost_estimate.get("estimated_total_cost", 0)

        if self.daily_cost + estimated_cost > self.config.cost_limit_daily:
            raise LLMError(f"Request would exceed daily cost limit: ${self.config.cost_limit_daily}")

        # Generate response
        try:
            response = await self.provider.generate(request)
            self.request_count += 1

            # Update cost tracking (use actual if available, estimate otherwise)
            if response.usage:
                actual_cost = self._calculate_actual_cost(response, request.model)
                self.daily_cost += actual_cost

                # Track persistent usage statistics
                usage_tracker = get_usage_tracker()
                usage_tracker.track_request(
                    cost=actual_cost,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    provider=self.config.provider
                )
            else:
                self.daily_cost += estimated_cost

                # Track with estimated cost
                usage_tracker = get_usage_tracker()
                usage_tracker.track_request(
                    cost=estimated_cost,
                    input_tokens=0,
                    output_tokens=0,
                    provider=self.config.provider
                )

            logger.info(f"LLM request completed. Total daily cost: ${self.daily_cost:.4f}")
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _calculate_actual_cost(self, response: LLMResponse, model: str) -> float:
        """Calculate actual cost from response usage."""
        if not response.usage:
            return 0.0

        model_info = self.provider.get_model_info(model)
        if not model_info:
            return 0.0

        input_cost = (response.usage.prompt_tokens / 1000) * model_info.get("input_cost_per_1k", 0)
        output_cost = (response.usage.completion_tokens / 1000) * model_info.get("output_cost_per_1k", 0)

        return input_cost + output_cost

    async def interpret_text(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Interpret text for structured extraction.

        Args:
            text: Text to interpret
            context: Additional context (current state, etc.)

        Returns:
            Interpretation result with structured data
        """
        system_prompt = """You are an expert at interpreting unstructured text input and extracting structured information for a productivity system.

Your task is to analyze the input and identify:
1. Projects (ongoing work streams, initiatives)
2. Tasks (actionable items with clear deliverables)
3. Notes (information for reference)
4. Promises/Commitments (things committed to others)

Respond with a JSON object containing:
- "projects": [{"name": "Project Name", "aliases": ["alias1"], "confidence": 0.9}]
- "tasks": [{"title": "Task Title", "project": "Project Name", "priority": "high|medium|low", "confidence": 0.8}]
- "type": "project|task|note|promise|mixed"
- "confidence": overall confidence score (0.0-1.0)
- "explanation": brief explanation of the interpretation

Be conservative with confidence scores. Only use high confidence (>0.8) when very certain."""

        context_info = ""
        if context:
            context_info = f"\n\nContext: {json.dumps(context, indent=2)}"

        prompt = f"""Interpret this input: "{text}"{context_info}"""

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1  # Low temperature for consistent interpretation
            )

            # Try to parse JSON response
            try:
                result = json.loads(response.content)
                result["llm_response"] = response
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                return {
                    "type": "unknown",
                    "confidence": 0.5,
                    "explanation": "Failed to parse structured response",
                    "raw_response": response.content,
                    "llm_response": response
                }

        except Exception as e:
            logger.error(f"Text interpretation failed: {e}")
            return {
                "type": "error",
                "confidence": 0.0,
                "explanation": f"Interpretation failed: {str(e)}",
                "error": str(e)
            }

    async def parse_temporal_expressions(
        self,
        text: str,
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Parse due dates and durations from text using LLM.

        Args:
            text: Text to analyze for temporal expressions
            current_date: Current date for relative date interpretation

        Returns:
            Dictionary with parsed temporal information
        """
        if current_date is None:
            current_date = datetime.now(UTC)

        system_prompt = f"""You are a temporal expression parser. Extract due dates and durations from natural language text.

Current context:
- Today's date: {current_date.strftime('%A, %B %d, %Y')} ({current_date.date().isoformat()})
- Current time: {current_date.strftime('%I:%M %p')}

Parse these temporal expressions:
- Due dates: "by Friday", "tomorrow", "Feb 15", "end of month", "in 3 days"
- Durations: "[30min]", "[2 hrs]", "[1.5h]", brackets indicate duration

Rules:
- "Friday" = next Friday from today
- "tomorrow" = next day
- "Feb 15" = Feb 15 of current/next year (whichever is sooner)
- Duration formats: [30min], [2 hrs], [1.5h], [20 minutes]
- If no explicit due date, set to null
- If no explicit duration, set to null

Respond with JSON:
{{
    "due_date": "ISO_FORMAT_WITH_TIME or null",
    "duration_minutes": integer_or_null,
    "confidence": float_between_0_and_1,
    "explanation": "brief explanation of interpretation"
}}

Be conservative with confidence. Only high confidence (>0.8) for very clear expressions."""

        prompt = f"""Parse temporal expressions from: "{text}"

Remember: Today is {current_date.strftime('%A, %B %d, %Y')}"""

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.1
            )

            # Parse JSON response, handling markdown formatting
            try:
                content = response.content.strip()

                # Remove markdown code block formatting if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()

                result = json.loads(content)

                # Validate and clean up the result
                if result.get("due_date") and result["due_date"] != "null":
                    try:
                        # Validate the ISO format
                        parsed_date = datetime.fromisoformat(result["due_date"].replace('Z', '+00:00'))
                        result["due_date"] = parsed_date.isoformat()
                    except (ValueError, AttributeError):
                        result["due_date"] = None
                        result["confidence"] = max(0.0, result.get("confidence", 0.0) - 0.2)
                else:
                    result["due_date"] = None

                # Validate duration
                if result.get("duration_minutes"):
                    try:
                        duration = int(result["duration_minutes"])
                        if duration <= 0 or duration > 24 * 60:  # Max 24 hours
                            result["duration_minutes"] = None
                            result["confidence"] = max(0.0, result.get("confidence", 0.0) - 0.1)
                        else:
                            result["duration_minutes"] = duration
                    except (ValueError, TypeError):
                        result["duration_minutes"] = None
                        result["confidence"] = max(0.0, result.get("confidence", 0.0) - 0.1)
                else:
                    result["duration_minutes"] = None

                # Ensure confidence is valid
                confidence = result.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)) or confidence < 0.0:
                    confidence = 0.5
                elif confidence > 1.0:
                    confidence = 1.0
                result["confidence"] = float(confidence)

                result["llm_response"] = response
                return result

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse temporal JSON response: {response.content}")
                return {
                    "due_date": None,
                    "duration_minutes": None,
                    "confidence": 0.0,
                    "explanation": "Failed to parse temporal response",
                    "raw_response": response.content,
                    "llm_response": response
                }

        except Exception as e:
            logger.error(f"Temporal parsing failed: {e}")
            return {
                "due_date": None,
                "duration_minutes": None,
                "confidence": 0.0,
                "explanation": f"Temporal parsing failed: {str(e)}",
                "error": str(e)
            }

    async def summarize_events(
        self,
        events: List[Dict[str, Any]],
        time_period: str = "week",
        style: str = "executive"
    ) -> Dict[str, Any]:
        """Generate summary of events.

        Args:
            events: List of event data
            time_period: Time period for summary (day, week, month)
            style: Summary style (executive, detailed, technical)

        Returns:
            Summary result
        """
        system_prompt = f"""You are an executive assistant creating {style} summaries of productivity data.

Create a {time_period}ly summary in {style} style. Focus on:
- Key accomplishments and outcomes
- Project progress and milestones
- Important tasks completed
- Areas needing attention
- Trends and insights

Keep the summary concise but informative. Use markdown formatting for readability."""

        events_text = json.dumps(events, indent=2, default=str)
        prompt = f"""Summarize these events for the past {time_period}:

{events_text}

Please provide a {style} summary in markdown format."""

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.3  # Slightly higher for more engaging summaries
            )

            return {
                "summary": response.content,
                "style": style,
                "time_period": time_period,
                "event_count": len(events),
                "llm_response": response
            }

        except Exception as e:
            logger.error(f"Event summarization failed: {e}")
            return {
                "summary": f"Summary generation failed: {str(e)}",
                "error": str(e),
                "style": style,
                "time_period": time_period,
                "event_count": len(events)
            }

    async def parse_task_update_request(
        self,
        natural_text: str,
        available_tasks: List[Dict[str, Any]],
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Parse natural language task update requests.

        Args:
            natural_text: Natural language update request
            available_tasks: List of current tasks with metadata
            current_date: Current date for temporal parsing

        Returns:
            Parsed update request with task match and changes
        """
        if current_date is None:
            current_date = datetime.now(UTC)

        # Create task context for LLM
        task_context = []
        for task in available_tasks[:20]:  # Limit to prevent token overflow
            task_context.append({
                "id_short": task.get("task_id_short", task.get("task_id", "")[:8]),  # Short ID for display
                "id_full": task.get("task_id", ""),  # Full ID for matching
                "title": task.get("title", ""),
                "project": task.get("project_name", ""),
                "priority": task.get("priority", "normal"),
                "status": task.get("status", "pending"),
                "due_date": task.get("scheduled_for", ""),
                "duration": task.get("duration_minutes", "")
            })

        system_prompt = f"""You are a task update parser. Parse natural language requests to update existing tasks.

Current context: Today's date: {current_date.strftime('%A, %B %d, %Y')}

Available tasks: {json.dumps(task_context, indent=2)}

Parse the request and identify:
1. Which task the user is referring to (match by keywords, project, or description)
2. What updates they want to make (priority, status, due date, duration)

Return JSON with this structure:
{{
    "task_matches": [
        {{
            "task_id": "full_task_id_from_id_full_field",
            "confidence": 0.9,
            "match_reason": "explanation of why this task matches"
        }}
    ],
    "updates": {{
        "priority": "high|normal|low|urgent|null",
        "status": "pending|in_progress|completed|cancelled|on_hold|null",
        "due_date": "ISO_FORMAT or relative like 'tomorrow'|null",
        "duration_minutes": integer_or_null
    }},
    "confidence": 0.9,
    "explanation": "explanation of the parsed request"
}}

Examples:
- "completed project update to Alice" → status: completed, match task about project/Alice
- "make xxx high priority and due date tomorrow" → priority: high, due_date: tomorrow
- "mark the email task as in progress" → status: in_progress, match email-related task

Be flexible with matching - use keywords, project names, or task descriptions. If multiple tasks could match, return them ranked by confidence."""

        try:
            response = await self.generate(
                prompt=f"Parse this task update request: '{natural_text}'",
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=800
            )

            # Parse JSON response
            try:
                content = response.content.strip()

                # Remove markdown code block formatting if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()

                result = json.loads(content)
                result["llm_response"] = response
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {response.content}")
                return {
                    "task_matches": [],
                    "updates": {},
                    "confidence": 0.0,
                    "error": "Failed to parse LLM response",
                    "raw_response": response.content
                }

        except Exception as e:
            logger.error(f"Task update parsing failed: {e}")
            return {
                "task_matches": [],
                "updates": {},
                "confidence": 0.0,
                "error": str(e)
            }

    async def parse_natural_query(
        self,
        question: str,
        available_tasks: List[Dict[str, Any]],
        available_projects: List[Dict[str, Any]],
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Parse natural language queries about tasks and projects.

        Args:
            question: Natural language question
            available_tasks: List of current tasks with metadata
            available_projects: List of current projects with metadata
            current_date: Current date for temporal parsing

        Returns:
            Parsed query with filters and response guidance
        """
        if current_date is None:
            current_date = datetime.now(UTC)

        # Create context for LLM
        task_context = []
        for task in available_tasks[:30]:  # More context for queries
            task_context.append({
                "id_short": task.get("task_id", "")[:8],
                "title": task.get("title", ""),
                "project": task.get("project_name", ""),
                "priority": task.get("priority", "normal"),
                "status": task.get("status", "pending"),
                "due_date": task.get("scheduled_for", ""),
                "duration": task.get("duration_minutes", ""),
                "created": task.get("created_at", "")
            })

        project_context = []
        for project in available_projects:
            project_context.append({
                "id": project.get("project_id", ""),
                "name": project.get("name", ""),
                "aliases": project.get("aliases", []),
                "task_count": project.get("task_count", 0)
            })

        system_prompt = f"""You are a task query parser. Parse natural language questions about tasks and projects into structured queries.

Current context: Today's date: {current_date.strftime('%A, %B %d, %Y')}

Available tasks: {json.dumps(task_context, indent=2)}
Available projects: {json.dumps(project_context, indent=2)}

Parse the question and return JSON with this structure:
{{
    "query_type": "list_tasks|count_tasks|show_projects|get_status",
    "filters": {{
        "project_id": "project_id_or_null",
        "project_name": "project_name_or_null",
        "priority": "high|normal|low|urgent|null",
        "status": "pending|in_progress|completed|cancelled|on_hold|null",
        "due_date_filter": "today|tomorrow|overdue|this_week|null",
        "created_filter": "today|this_week|null",
        "keywords": ["keyword1", "keyword2"]
    }},
    "response_style": "conversational|tabular|summary",
    "confidence": 0.9,
    "explanation": "explanation of the parsed query"
}}

Examples:
- "what is due tomorrow?" → query_type: list_tasks, due_date_filter: tomorrow
- "show me ca zap tasks" → query_type: list_tasks, keywords: ["ca", "zap"]
- "how many high priority tasks?" → query_type: count_tasks, priority: high
- "what's overdue?" → query_type: list_tasks, due_date_filter: overdue
- "show me lpd project tasks" → query_type: list_tasks, project_name: "lpd"
- "what am I working on?" → query_type: list_tasks, status: in_progress

Be flexible with project matching - match by name, aliases, or similar keywords."""

        try:
            response = await self.generate(
                prompt=f"Parse this question: '{question}'",
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=600
            )

            # Parse JSON response
            try:
                content = response.content.strip()

                # Remove markdown code block formatting if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()

                result = json.loads(content)
                result["llm_response"] = response
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {response.content}")
                return {
                    "query_type": "list_tasks",
                    "filters": {},
                    "response_style": "conversational",
                    "confidence": 0.0,
                    "error": "Failed to parse LLM response",
                    "raw_response": response.content
                }

        except Exception as e:
            logger.error(f"Natural query parsing failed: {e}")
            return {
                "query_type": "list_tasks",
                "filters": {},
                "response_style": "conversational",
                "confidence": 0.0,
                "error": str(e)
            }

    async def parse_mixed_content(
        self,
        text: str,
        current_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Parse natural language input that may contain tasks, artifacts, and relationships.

        This is the core Phase 7 functionality that enables intelligent extraction of
        both tasks and artifacts from complex natural language input.

        Args:
            text: Natural language input that may contain mixed content
            current_context: Optional context including current projects, tasks, etc.

        Returns:
            Dictionary with parsed tasks, artifacts, relationships, and metadata
        """
        from datetime import datetime, UTC

        current_date = datetime.now(UTC)

        # Prepare context information for better parsing
        context_info = ""
        if current_context:
            projects = current_context.get('projects', {})
            if projects:
                project_names = list(projects.keys())[:10]  # Limit to prevent token overflow
                context_info += f"\nActive projects: {', '.join(project_names)}"

            tasks = current_context.get('tasks', {})
            if tasks:
                active_tasks = [task for task in tasks.values() if task.get('status') != 'completed'][:10]
                if active_tasks:
                    task_summaries = [f"'{task.get('title', '')[:30]}...'" for task in active_tasks]
                    context_info += f"\nActive tasks: {', '.join(task_summaries)}"

        system_prompt = f"""You are an intelligent content parser that extracts tasks, artifacts, and their relationships from natural language input.

Current context:
- Today's date: {current_date.strftime('%A, %B %d, %Y')} ({current_date.date().isoformat()})
- Current time: {current_date.strftime('%I:%M %p')}{context_info}

Your task is to analyze natural language input and identify:

1. **Tasks** - Actionable items, follow-ups, TODOs, work items
2. **Artifacts** - External references like emails, Slack messages, documents, meeting notes, calls
3. **Relationships** - How tasks and artifacts relate to each other
4. **Project associations** - Which projects this content relates to

Types of artifacts to detect:
- **slack_msg**: Slack messages, DMs, channel posts (look for "slack", "sent message", "DM")
- **email**: Email content, correspondence (look for "email", "sent to", "received from")
- **doc**: Documents, files, links to SharePoint/Google Drive/etc (look for URLs, "document", "shared doc")
- **meeting_notes**: Meeting summaries, standup notes (look for "meeting", "discussed", "standup")
- **call_notes**: Phone calls, video calls (look for "call", "spoke with", "discussed on call")
- **quip**: Quip documents (look for quip.com URLs)
- **sharepoint**: SharePoint documents (look for sharepoint.com URLs)
- **gdrive**: Google Drive files (look for drive.google.com URLs)

Examples of mixed content:
- "Sent project slack message to Alice: 'Progress update...' - need to follow up by Friday"
  → Task: "Follow up with Alice by Friday", Artifact: Slack message with content
- "Got approval email from legal team - can proceed with migration"
  → Task: "Proceed with migration", Artifact: Email from legal team
- "Meeting notes from team standup: discussed blockers and next steps"
  → Artifact: Meeting notes, possibly Task: address discussed items

Return JSON with this exact structure:
{{
    "tasks": [
        {{
            "title": "Clear, actionable task title",
            "description": "Optional detailed description",
            "priority": "high|normal|low|urgent|null",
            "status": "pending|in_progress|completed|null",
            "due_date": "ISO_FORMAT or 'tomorrow'|'friday'|null",
            "duration_minutes": integer_or_null,
            "project_hints": ["keyword1", "keyword2"],
            "confidence": 0.9
        }}
    ],
    "artifacts": [
        {{
            "artifact_type": "slack_msg|email|doc|meeting_notes|call_notes|quip|sharepoint|gdrive",
            "title": "Human-readable title for the artifact",
            "content": "Full embedded content if present, null otherwise",
            "url": "Extracted URL if present, null otherwise",
            "source": "Source identifier like 'slack:channel.timestamp'",
            "project_hints": ["keyword1", "keyword2"],
            "task_hints": ["relates to task by..."],
            "metadata": {{"participants": ["person1"], "channel": "name"}},
            "confidence": 0.8
        }}
    ],
    "relationships": [
        {{
            "relationship_type": "task_artifact_link",
            "task_index": 0,
            "artifact_index": 0,
            "description": "Artifact supports/relates to task",
            "confidence": 0.9
        }}
    ],
    "overall_confidence": 0.85,
    "explanation": "Clear explanation of what was parsed and why",
    "project_suggestions": ["project1", "project2"],
    "parsing_method": "mixed_content_llm"
}}

Guidelines:
- Be conservative with confidence scores (>0.8 only when very certain)
- Extract full content when explicitly provided (e.g. "sent this email: [content]")
- Generate descriptive titles for artifacts
- Use project_hints to suggest associations (e.g. "project-x", "dashboard", "q1-planning")
- Create relationships when tasks and artifacts clearly relate
- If input is just a task or just an artifact, still return the full structure
- Use null for optional fields when not applicable

Focus on being intelligent about context - if someone mentions "sent slack message about project X", that's likely both an artifact (the message) and possibly a task (follow up)."""

        prompt = f"""Parse this mixed content input: "{text}"

Remember: Today is {current_date.strftime('%A, %B %d, %Y')}"""

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=5000   # Generous limit for comprehensive parsing of long content
            )

            # Parse JSON response
            try:
                content = response.content.strip()

                # Remove markdown code block formatting if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()

                result = json.loads(content)

                # Validate and clean up the result
                self._validate_parsed_content(result)

                result["llm_response"] = response
                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse mixed content JSON response at position {e.pos}: {str(e)}")
                logger.error(f"Response length: {len(response.content)} characters")
                logger.error(f"Response preview: {response.content[:500]}...")
                logger.error(f"Response ending: ...{response.content[-200:]}")

                # Check if response was likely truncated
                was_truncated = not response.content.strip().endswith('}')
                truncation_hint = " (Response appears truncated - consider shorter input or higher token limit)" if was_truncated else ""

                return {
                    "tasks": [],
                    "artifacts": [],
                    "relationships": [],
                    "overall_confidence": 0.0,
                    "explanation": f"JSON parsing failed at position {e.pos}: {str(e)}{truncation_hint}",
                    "project_suggestions": [],
                    "parsing_method": "mixed_content_llm_failed",
                    "error": "JSON parsing failed",
                    "json_error_position": e.pos,
                    "response_length": len(response.content),
                    "likely_truncated": was_truncated,
                    "raw_response": response.content
                }

        except Exception as e:
            logger.error(f"Mixed content parsing failed: {e}")
            return {
                "tasks": [],
                "artifacts": [],
                "relationships": [],
                "overall_confidence": 0.0,
                "explanation": f"Mixed content parsing failed: {str(e)}",
                "project_suggestions": [],
                "parsing_method": "mixed_content_llm",
                "error": str(e)
            }

    def _validate_parsed_content(self, result: Dict[str, Any]) -> None:
        """Validate and clean up parsed content result."""
        # Ensure required fields exist
        result.setdefault("tasks", [])
        result.setdefault("artifacts", [])
        result.setdefault("relationships", [])
        result.setdefault("overall_confidence", 0.5)
        result.setdefault("explanation", "No explanation provided")
        result.setdefault("project_suggestions", [])
        result.setdefault("parsing_method", "mixed_content_llm")

        # Validate confidence scores
        if not isinstance(result["overall_confidence"], (int, float)):
            result["overall_confidence"] = 0.5
        result["overall_confidence"] = max(0.0, min(1.0, float(result["overall_confidence"])))

        # Validate tasks
        for task in result["tasks"]:
            if not isinstance(task.get("confidence", 0), (int, float)):
                task["confidence"] = 0.5
            task["confidence"] = max(0.0, min(1.0, float(task.get("confidence", 0.5))))

            # Ensure required fields
            task.setdefault("title", "Untitled task")
            task.setdefault("project_hints", [])

            # Validate priority and status
            valid_priorities = ["high", "normal", "low", "urgent", None]
            if task.get("priority") not in valid_priorities:
                task["priority"] = None

            valid_statuses = ["pending", "in_progress", "completed", "cancelled", "on_hold", None]
            if task.get("status") not in valid_statuses:
                task["status"] = None

        # Validate artifacts
        for artifact in result["artifacts"]:
            if not isinstance(artifact.get("confidence", 0), (int, float)):
                artifact["confidence"] = 0.5
            artifact["confidence"] = max(0.0, min(1.0, float(artifact.get("confidence", 0.5))))

            # Ensure required fields
            artifact.setdefault("title", "Untitled artifact")
            artifact.setdefault("artifact_type", "doc")  # Default fallback
            artifact.setdefault("project_hints", [])
            artifact.setdefault("task_hints", [])
            artifact.setdefault("metadata", {})

        # Validate relationships
        for relationship in result["relationships"]:
            if not isinstance(relationship.get("confidence", 0), (int, float)):
                relationship["confidence"] = 0.5
            relationship["confidence"] = max(0.0, min(1.0, float(relationship.get("confidence", 0.5))))

            # Ensure indices are valid
            task_index = relationship.get("task_index", -1)
            artifact_index = relationship.get("artifact_index", -1)

            if (not isinstance(task_index, int) or task_index < 0 or
                task_index >= len(result["tasks"]) or
                not isinstance(artifact_index, int) or artifact_index < 0 or
                artifact_index >= len(result["artifacts"])):
                # Invalid relationship, mark for removal
                relationship["_invalid"] = True
            else:
                relationship.setdefault("relationship_type", "task_artifact_link")
                relationship.setdefault("description", "Related items")

        # Remove invalid relationships
        result["relationships"] = [
            rel for rel in result["relationships"]
            if not rel.get("_invalid", False)
        ]

    async def analyze_question_intent(
        self,
        question: str,
        available_commands: List[str],
        system_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a natural language question to determine intent and create execution plan.

        Args:
            question: Natural language question from user
            available_commands: List of available command functions
            system_context: Current system state context

        Returns:
            Dict with intent analysis and execution plan
        """
        from datetime import datetime, UTC

        current_date = datetime.now(UTC)

        # Prepare context for LLM
        context_info = ""
        if system_context:
            context_info = f"""
System Context:
- Projects: {len(system_context.get('projects', {}))} projects available
- Tasks: {len(system_context.get('active_tasks', []))} active, {len(system_context.get('completed_tasks', []))} completed
- Artifacts: {len(system_context.get('artifacts', {}))} artifacts available
- Current focus: {system_context.get('current_focus_project', 'None')}
"""

        system_prompt = f"""You are an intelligent assistant that analyzes user questions and creates execution plans using available system commands.

Current date: {current_date.strftime('%A, %B %d, %Y')}
{context_info}

Available commands you can use:
{', '.join(available_commands)}

Analyze the user's question and determine:
1. What type of information they're seeking
2. Which commands/functions would best answer their question
3. What parameters those commands need
4. Fallback strategies if primary approach fails

Return JSON with this structure:
{{
    "intent": {{
        "category": "artifacts|tasks|projects|status|summary|mixed",
        "subcategory": "list|show|filter|analyze|count",
        "entity": "project_name_or_null",
        "scope": "today|this_week|all|specific_date",
        "confidence": 0.9
    }},
    "execution_plan": [
        {{
            "command": "command_name",
            "parameters": {{"param1": "value1"}},
            "description": "what this step accomplishes",
            "fallback": "alternative_command_if_this_fails"
        }}
    ],
    "response_guidance": {{
        "format": "list|summary|detailed|count",
        "tone": "conversational|technical|brief",
        "highlight": ["key", "points", "to", "emphasize"]
    }},
    "confidence": 0.9,
    "reasoning": "explanation of analysis"
}}

IMPORTANT: When extracting project names from questions:
- Remove words like "project", "work", "tasks" from project names
- Extract just the core project identifier
- Use the exact case and format from the user's question

Examples:
- "Show me artifacts for NA Cube project" → category: artifacts, command: get_project_artifacts, parameters: {{"project": "NA Cube"}}
- "NA Cube project artifacts" → parameters: {{"project": "NA Cube"}}
- "What tasks are due today?" → category: tasks, command: get_tasks_due_today
- "How many completed tasks do I have?" → category: tasks, subcategory: count, command: count_completed_tasks"""

        try:
            response = await self.generate(
                prompt=f"Analyze this question and create execution plan: '{question}'",
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1000
            )

            # Parse JSON response
            try:
                content = response.content.strip()

                # Remove markdown code block formatting if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                result = json.loads(content)
                result["llm_response"] = response
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from intent analysis: {response.content}")
                return {
                    "intent": {"category": "unknown", "confidence": 0.0},
                    "execution_plan": [],
                    "confidence": 0.0,
                    "error": "Failed to parse LLM response",
                    "raw_response": response.content
                }

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "intent": {"category": "unknown", "confidence": 0.0},
                "execution_plan": [],
                "confidence": 0.0,
                "error": str(e)
            }

    async def synthesize_response(
        self,
        original_question: str,
        execution_results: List[Dict[str, Any]],
        intent_analysis: Dict[str, Any]
    ) -> str:
        """Synthesize a natural language response from execution results.

        Args:
            original_question: The user's original question
            execution_results: Results from executing the planned commands
            intent_analysis: The original intent analysis

        Returns:
            Natural language response string
        """

        # Prepare results summary for LLM
        results_summary = []
        for i, result in enumerate(execution_results):
            command = result.get('command', 'unknown')
            success = result.get('success', False)
            data = result.get('data')
            error = result.get('error')

            if success and data:
                if isinstance(data, list):
                    results_summary.append(f"Command {i+1} ({command}): Found {len(data)} items")
                elif isinstance(data, dict):
                    results_summary.append(f"Command {i+1} ({command}): Retrieved data with {len(data)} fields")
                else:
                    results_summary.append(f"Command {i+1} ({command}): Retrieved: {str(data)[:100]}")
            else:
                results_summary.append(f"Command {i+1} ({command}): Failed - {error}")

        response_guidance = intent_analysis.get('response_guidance', {})
        format_type = response_guidance.get('format', 'conversational')
        tone = response_guidance.get('tone', 'conversational')

        system_prompt = f"""You are a helpful assistant that presents information in response to user questions.

The user asked: "{original_question}"

Command execution results:
{chr(10).join(results_summary)}

Raw execution data: {json.dumps(execution_results, indent=2, default=str)}

Create a {tone} response in {format_type} format. Guidelines:
- Answer the user's question directly and completely
- Present information clearly and concisely
- If no results were found, explain why and suggest alternatives
- Use appropriate emojis sparingly (📎 for artifacts, ✅ for tasks, 📂 for projects)
- Focus on the most relevant information first
- If there were errors, mention them but don't make them the focus

Response style: {tone}
Format preference: {format_type}"""

        try:
            response = await self.generate(
                prompt="Create a helpful response based on the execution results.",
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=800
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Fallback to basic response
            if execution_results and any(r.get('success') for r in execution_results):
                return "I found some information for your question, but had trouble formatting the response. Please try a more specific query."
            else:
                return "I wasn't able to find the information you requested. Please try rephrasing your question or being more specific."

    def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "request_count": self.request_count,
            "daily_cost": self.daily_cost,
            "cost_limit": self.config.cost_limit_daily,
            "available_models": self.provider.list_available_models() if self.provider else [],
            "config": self.config.to_dict()
        }

    def reset_daily_cost(self) -> None:
        """Reset daily cost counter."""
        self.daily_cost = 0.0
        logger.info("Daily cost counter reset")

    def save_config(self, config_path: Optional[Path] = None) -> None:
        """Save current configuration to file.

        Args:
            config_path: Path to save config (default: ~/.config/sidecar/llm.json)
        """
        if config_path is None:
            config_path = Path.home() / ".config" / "sidecar" / "llm.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {config_path}")