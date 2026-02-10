"""LLM service orchestrator and configuration management."""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

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
            else:
                self.daily_cost += estimated_cost

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