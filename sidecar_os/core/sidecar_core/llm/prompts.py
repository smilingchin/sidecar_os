"""Prompt templates for LLM interactions."""

from typing import Dict, Any, List, Optional


class PromptTemplates:
    """Collection of reusable prompt templates for different LLM tasks."""

    @staticmethod
    def text_interpretation_system_prompt() -> str:
        """System prompt for text interpretation tasks."""
        return """You are an expert at interpreting unstructured text input and extracting structured information for a productivity system called Sidecar OS.

Your task is to analyze user input and identify:

1. **Projects**: Ongoing work streams, initiatives, or areas of focus
   - Look for: project names, acronyms (LPD, EB1), explicit project mentions
   - Consider: context clues, domain-specific terms

2. **Tasks**: Actionable items with clear deliverables
   - Look for: action verbs (need to, should, must, implement, call, review)
   - Consider: urgency indicators, deadlines, specific actions

3. **Notes**: Information for reference without immediate action
   - Look for: status updates, observations, meeting notes
   - Consider: informational content, context sharing

4. **Promises/Commitments**: Things committed to others or with deadlines
   - Look for: "will", "by [date]", "promised", commitments to others
   - Consider: external dependencies, accountability items

**Response Format**: Always respond with valid JSON containing:
```json
{
  "projects": [
    {
      "name": "Project Name",
      "aliases": ["alias1", "alias2"],
      "confidence": 0.9,
      "reasoning": "why this is a project"
    }
  ],
  "tasks": [
    {
      "title": "Clear task title",
      "project": "Associated project name or null",
      "priority": "high|medium|low",
      "confidence": 0.8,
      "reasoning": "why this is a task"
    }
  ],
  "type": "project|task|note|promise|mixed",
  "overall_confidence": 0.85,
  "explanation": "Brief explanation of the interpretation"
}
```

**Confidence Guidelines**:
- 0.9+: Very certain (clear action verbs, explicit project names)
- 0.7-0.9: High confidence (strong indicators, context clues)
- 0.5-0.7: Medium confidence (some indicators, requires inference)
- 0.3-0.5: Low confidence (ambiguous, unclear intent)
- 0.0-0.3: Very uncertain (insufficient information)

Be conservative with confidence scores. Only use high confidence when very certain."""

    @staticmethod
    def interpretation_user_prompt(text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """User prompt for text interpretation with optional context.

        Args:
            text: Text to interpret
            context: Additional context (projects, recent activity, etc.)

        Returns:
            Formatted prompt
        """
        prompt = f'Interpret this input: "{text}"'

        if context:
            context_parts = []

            if context.get("current_focus_project"):
                context_parts.append(f"Current focus: {context['current_focus_project']}")

            if context.get("recent_projects"):
                projects = ", ".join(context["recent_projects"])
                context_parts.append(f"Recent projects: {projects}")

            if context.get("active_tasks_count"):
                context_parts.append(f"Active tasks: {context['active_tasks_count']}")

            if context.get("recent_activity"):
                context_parts.append(f"Recent activity: {context['recent_activity']}")

            if context_parts:
                prompt += f"\n\nContext:\n" + "\n".join(f"- {part}" for part in context_parts)

        return prompt

    @staticmethod
    def weekly_summary_system_prompt(style: str = "executive") -> str:
        """System prompt for weekly summary generation.

        Args:
            style: Summary style (executive, detailed, technical)

        Returns:
            System prompt for the style
        """
        if style == "executive":
            return """You are an executive assistant creating concise, high-level weekly summaries for busy professionals.

Focus on:
- **Key Accomplishments**: Major wins and completed deliverables
- **Project Progress**: Significant milestones and status updates
- **Strategic Insights**: Patterns, trends, and important observations
- **Priorities Ahead**: Critical items requiring attention next week

Style Guidelines:
- Use clear, business-appropriate language
- Emphasize outcomes over activities
- Keep it concise but informative (200-400 words)
- Use markdown formatting for readability
- Lead with the most important items"""

        elif style == "detailed":
            return """You are a productivity assistant creating comprehensive weekly summaries with full context and details.

Focus on:
- **Complete Activity Breakdown**: All projects, tasks, and activities
- **Detailed Progress**: Specific accomplishments and status updates
- **Context and Background**: Relevant details and decision points
- **Comprehensive Planning**: Detailed next steps and dependencies

Style Guidelines:
- Include specific details and context
- Provide complete picture of the week's work
- Use structured format with clear sections
- Include quantitative metrics when available
- Aim for 400-800 words with full context"""

        elif style == "technical":
            return """You are a technical assistant creating detailed summaries focused on implementation, metrics, and technical progress.

Focus on:
- **Technical Achievements**: Code, experiments, implementations
- **Metrics and Data**: Quantitative results and measurements
- **System Changes**: Infrastructure, tools, process improvements
- **Technical Challenges**: Blockers, technical debt, solutions

Style Guidelines:
- Use precise technical language
- Include specific metrics, numbers, and measurements
- Focus on technical details and implementation
- Highlight technical decisions and rationale
- Structure with clear technical categories"""

        else:
            return """You are a helpful assistant creating weekly summaries. Adapt your style based on the content and context provided."""

    @staticmethod
    def weekly_summary_user_prompt(
        events: List[Dict[str, Any]],
        time_period: str = "week",
        additional_context: str = ""
    ) -> str:
        """User prompt for weekly summary generation.

        Args:
            events: Event data to summarize
            time_period: Time period (week, month, etc.)
            additional_context: Additional context or instructions

        Returns:
            Formatted prompt
        """
        prompt = f"Create a {time_period}ly summary based on these events and activities:\n\n"

        # Group events by type for better organization
        projects = []
        tasks = []
        inbox_items = []
        other_events = []

        for event in events:
            event_type = event.get("event_type", "unknown")
            if "project" in event_type:
                projects.append(event)
            elif "task" in event_type:
                tasks.append(event)
            elif "inbox" in event_type:
                inbox_items.append(event)
            else:
                other_events.append(event)

        # Add organized event data
        if projects:
            prompt += f"## Project Events ({len(projects)} items)\n"
            for event in projects[:10]:  # Limit for prompt size
                prompt += f"- {event.get('timestamp', 'Unknown time')}: {event.get('event_type', 'Unknown')}\n"
                if event.get('payload'):
                    payload = event['payload']
                    if payload.get('name'):
                        prompt += f"  Project: {payload['name']}\n"
            prompt += "\n"

        if tasks:
            prompt += f"## Task Events ({len(tasks)} items)\n"
            for event in tasks[:15]:  # Show more tasks
                prompt += f"- {event.get('timestamp', 'Unknown time')}: {event.get('event_type', 'Unknown')}\n"
                if event.get('payload'):
                    payload = event['payload']
                    if payload.get('title'):
                        prompt += f"  Task: {payload['title']}\n"
            prompt += "\n"

        if inbox_items:
            prompt += f"## Captured Items ({len(inbox_items)} items)\n"
            for event in inbox_items[:10]:
                prompt += f"- {event.get('timestamp', 'Unknown time')}: {event.get('event_type', 'Unknown')}\n"
                if event.get('payload', {}).get('text'):
                    prompt += f"  Text: {event['payload']['text'][:100]}...\n"
            prompt += "\n"

        if other_events:
            prompt += f"## Other Activities ({len(other_events)} items)\n"
            for event in other_events[:5]:
                prompt += f"- {event.get('timestamp', 'Unknown time')}: {event.get('event_type', 'Unknown')}\n"
            prompt += "\n"

        if additional_context:
            prompt += f"## Additional Context\n{additional_context}\n\n"

        prompt += f"Please analyze these events and create a comprehensive {time_period}ly summary using the specified style and format."

        return prompt

    @staticmethod
    def clarification_prompt(
        original_text: str,
        ambiguity_reason: str,
        suggested_interpretations: List[str]
    ) -> str:
        """Generate clarification request prompt.

        Args:
            original_text: Original ambiguous text
            ambiguity_reason: Why clarification is needed
            suggested_interpretations: Possible interpretations

        Returns:
            Clarification prompt
        """
        prompt = f"""The input "{original_text}" is ambiguous and needs clarification.

**Ambiguity reason**: {ambiguity_reason}

**Possible interpretations**:"""

        for i, interpretation in enumerate(suggested_interpretations, 1):
            prompt += f"\n{i}. {interpretation}"

        prompt += "\n\nPlease help clarify the intent by providing specific questions that would resolve the ambiguity."

        return prompt

    @staticmethod
    def get_prompt_by_name(name: str, **kwargs) -> str:
        """Get a prompt template by name with parameters.

        Args:
            name: Prompt template name
            **kwargs: Template parameters

        Returns:
            Formatted prompt

        Raises:
            ValueError: If prompt template not found
        """
        templates = {
            "text_interpretation_system": PromptTemplates.text_interpretation_system_prompt,
            "interpretation_user": lambda: PromptTemplates.interpretation_user_prompt(
                kwargs.get("text", ""), kwargs.get("context")
            ),
            "weekly_summary_system": lambda: PromptTemplates.weekly_summary_system_prompt(
                kwargs.get("style", "executive")
            ),
            "weekly_summary_user": lambda: PromptTemplates.weekly_summary_user_prompt(
                kwargs.get("events", []),
                kwargs.get("time_period", "week"),
                kwargs.get("additional_context", "")
            ),
            "clarification": lambda: PromptTemplates.clarification_prompt(
                kwargs.get("original_text", ""),
                kwargs.get("ambiguity_reason", ""),
                kwargs.get("suggested_interpretations", [])
            ),
        }

        if name not in templates:
            raise ValueError(f"Unknown prompt template: {name}")

        return templates[name]()