"""Mock LLM provider for testing and development."""

import asyncio
import random
from typing import Dict, List, Any
from datetime import datetime, UTC

from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMRole,
)


class MockProvider(LLMProvider):
    """Mock LLM provider that generates predictable responses for testing."""

    def __init__(self, delay: float = 0.1, fail_rate: float = 0.0):
        """Initialize mock provider.

        Args:
            delay: Artificial delay to simulate API latency
            fail_rate: Rate of random failures (0.0 to 1.0)
        """
        self.delay = delay
        self.fail_rate = fail_rate
        self.request_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response.

        Args:
            request: LLM request

        Returns:
            Mock LLM response
        """
        self.validate_request(request)
        self.request_count += 1

        # Simulate API delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Simulate random failures
        if self.fail_rate > 0 and random.random() < self.fail_rate:
            raise Exception("Mock provider random failure")

        # Generate response based on request content
        response_content = self._generate_response_content(request)

        # Calculate mock usage
        input_tokens = sum(len(msg.content.split()) for msg in request.messages)
        output_tokens = len(response_content.split())

        usage = LLMUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

        return LLMResponse(
            content=response_content,
            model=request.model,
            usage=usage,
            finish_reason="stop",
            request_id=f"mock-{self.request_count:06d}",
            created_at=datetime.now(UTC),
            metadata={
                "provider": "mock",
                "request_count": self.request_count,
            }
        )

    def _generate_response_content(self, request: LLMRequest) -> str:
        """Generate mock response content based on request.

        Args:
            request: LLM request

        Returns:
            Generated response content
        """
        # Get the last user message
        user_messages = [msg for msg in request.messages if msg.role == LLMRole.USER]
        if not user_messages:
            return "I need a user message to respond to."

        last_message = user_messages[-1].content.lower()

        # Pattern-based responses for common Sidecar use cases
        if "project" in last_message and ("create" in last_message or "new" in last_message):
            return self._mock_project_creation_response(last_message)
        elif "task" in last_message and ("create" in last_message or "add" in last_message):
            return self._mock_task_creation_response(last_message)
        elif "summary" in last_message or "summarize" in last_message:
            return self._mock_summary_response(last_message)
        elif "interpret" in last_message or "analyze" in last_message:
            return self._mock_interpretation_response(last_message)
        else:
            return self._mock_generic_response(last_message)

    def _mock_project_creation_response(self, message: str) -> str:
        """Mock response for project creation requests."""
        if "lpd" in message.lower():
            return '''Based on the context, I can see this relates to an LPD (likely machine learning) project. Here's my interpretation:

**Project**: LPD Experiments
**Description**: Machine learning project involving ablation studies and model experimentation
**Suggested aliases**: ["lpd", "ml-experiments"]

This appears to be a high-confidence project match (95%).'''

        elif "eb1" in message.lower():
            return '''This appears to relate to an EB1 immigration petition project. Here's my interpretation:

**Project**: EB1 Petition Process
**Description**: Immigration petition management and coordination
**Suggested aliases**: ["eb1", "petition", "immigration"]

High confidence project identification (90%).'''

        else:
            return '''I can identify this as a project creation request, but need more context to provide specific recommendations.

**Confidence**: Medium (70%)
**Suggestion**: Please provide more details about the project domain or objectives.'''

    def _mock_task_creation_response(self, message: str) -> str:
        """Mock response for task creation requests."""
        if "call" in message:
            return '''**Task identified**: Communication/Follow-up task
**Action**: Call or contact someone
**Priority**: Medium
**Confidence**: 85%

This appears to be a clear action item requiring outreach or communication.'''

        elif "run" in message or "execute" in message:
            return '''**Task identified**: Execution/Implementation task
**Action**: Run experiments or execute process
**Priority**: High (execution tasks often time-sensitive)
**Confidence**: 90%

This is a clear implementation task.'''

        else:
            return '''**Task identified**: General task
**Confidence**: 70%
**Recommendation**: Consider adding more specific action verbs for better categorization.'''

    def _mock_summary_response(self, message: str) -> str:
        """Mock response for summary requests."""
        return '''# Weekly Summary - Mock Data

## Key Accomplishments
- 3 LPD experiments completed with 6pp improvement
- EB1 petition documentation submitted to attorney
- 2 new projects initiated

## Active Projects
- **LPD Experiments**: 4 active tasks, 2 completed this week
- **EB1 Petition**: 1 active task, awaiting attorney response

## Upcoming Priorities
- Review LPD experiment results
- Follow up on petition timeline
- Plan next week's research priorities

*This is a mock summary for demonstration purposes.*'''

    def _mock_interpretation_response(self, message: str) -> str:
        """Mock response for interpretation requests."""
        return '''**Input Analysis**: I can identify structured elements in this input.

**Detected Patterns**:
- Project indicators: Found potential project references
- Action items: Identified 2-3 actionable tasks
- Context clues: Medium confidence on categorization

**Recommendations**:
1. Create/focus on identified project
2. Generate structured tasks for action items
3. Tag for follow-up if ambiguous

**Overall Confidence**: 75%'''

    def _mock_generic_response(self, message: str) -> str:
        """Generic mock response."""
        return f'''I understand you're asking about: "{message[:50]}..."

This is a mock response from the LLM provider. In a real implementation, I would:
- Analyze the context and intent
- Provide structured recommendations
- Generate appropriate confidence scores

Mock provider is functioning correctly.'''

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "name": model_name,
            "provider": "mock",
            "max_tokens": 4000,
            "cost_per_1k_tokens": 0.001,  # Very cheap for testing
            "description": f"Mock model: {model_name}"
        }

    def list_available_models(self) -> List[str]:
        """List mock models."""
        return [
            "mock-claude-3-opus",
            "mock-claude-3-sonnet",
            "mock-claude-3-haiku",
            "mock-gpt-4",
        ]

    def estimate_cost(self, request: LLMRequest) -> Dict[str, float]:
        """Estimate mock cost."""
        input_tokens = sum(len(msg.content.split()) for msg in request.messages)
        output_tokens = request.max_tokens or 1000

        return {
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_input_cost": input_tokens * 0.001 / 1000,
            "estimated_output_cost": output_tokens * 0.001 / 1000,
            "estimated_total_cost": (input_tokens + output_tokens) * 0.001 / 1000,
        }