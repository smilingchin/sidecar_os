"""Base LLM provider interface and data models."""

from abc import ABC, abstractmethod
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMRole(Enum):
    """Message roles in LLM conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""
    role: LLMRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    messages: List[LLMMessage]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMUsage:
    """Token usage information from LLM response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: Optional[LLMUsage] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    created_at: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded error."""
    pass


class LLMAuthenticationError(LLMError):
    """Authentication error."""
    pass


class LLMServiceError(LLMError):
    """Service unavailable error."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            request: The LLM request with messages and parameters

        Returns:
            LLM response with content and metadata

        Raises:
            LLMError: Various LLM-related errors
        """
        pass

    @abstractmethod
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """List all available models from this provider.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> Dict[str, float]:
        """Estimate the cost of a request.

        Args:
            request: The LLM request

        Returns:
            Dictionary with cost estimates
        """
        pass

    def validate_request(self, request: LLMRequest) -> None:
        """Validate an LLM request.

        Args:
            request: The request to validate

        Raises:
            ValueError: If request is invalid
        """
        if not request.messages:
            raise ValueError("Request must contain at least one message")

        if not request.model:
            raise ValueError("Request must specify a model")

        for msg in request.messages:
            if not msg.content.strip():
                raise ValueError("Message content cannot be empty")

        if request.max_tokens is not None and request.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")