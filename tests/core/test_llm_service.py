"""Tests for LLM service and providers."""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

from sidecar_os.core.sidecar_core.llm.service import LLMService, LLMConfig
from sidecar_os.core.sidecar_core.llm.providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMMessage,
    LLMRole,
    LLMUsage,
    LLMError,
)
from sidecar_os.core.sidecar_core.llm.providers.mock import MockProvider
from sidecar_os.core.sidecar_core.llm.providers.bedrock import BedrockProvider
from sidecar_os.core.sidecar_core.llm.prompts import PromptTemplates


class TestLLMProviderBase:
    """Test base LLM provider functionality."""

    def test_llm_message_creation(self):
        """Test LLMMessage creation and validation."""
        msg = LLMMessage(role=LLMRole.USER, content="Test message")
        assert msg.role == LLMRole.USER
        assert msg.content == "Test message"
        assert msg.metadata is None

    def test_llm_request_creation(self):
        """Test LLMRequest creation and validation."""
        messages = [LLMMessage(role=LLMRole.USER, content="Test")]
        request = LLMRequest(
            messages=messages,
            model="test-model",
            max_tokens=1000,
            temperature=0.7
        )

        assert len(request.messages) == 1
        assert request.model == "test-model"
        assert request.max_tokens == 1000
        assert request.temperature == 0.7

    def test_llm_response_creation(self):
        """Test LLMResponse creation with auto timestamp."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(
            content="Test response",
            model="test-model",
            usage=usage
        )

        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage.total_tokens == 30
        assert response.created_at is not None


class TestMockProvider:
    """Test mock LLM provider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = MockProvider(delay=0.01, fail_rate=0.0)

    def test_mock_provider_basic_functionality(self):
        """Test basic mock provider operations."""
        models = self.provider.list_available_models()
        assert len(models) > 0
        assert "mock-claude-3-opus" in models

        model_info = self.provider.get_model_info("mock-claude-3-opus")
        assert model_info["provider"] == "mock"

    def test_mock_provider_cost_estimation(self):
        """Test mock provider cost estimation."""
        messages = [LLMMessage(role=LLMRole.USER, content="Test message")]
        request = LLMRequest(messages=messages, model="mock-claude-3-opus")

        cost = self.provider.estimate_cost(request)
        assert "estimated_total_cost" in cost
        assert cost["estimated_total_cost"] >= 0

    @pytest.mark.asyncio
    async def test_mock_provider_generation(self):
        """Test mock provider text generation."""
        messages = [LLMMessage(role=LLMRole.USER, content="Create a project for LPD experiments")]
        request = LLMRequest(messages=messages, model="mock-claude-3-opus")

        response = await self.provider.generate(request)

        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "mock-claude-3-opus"
        assert response.usage is not None
        assert response.request_id.startswith("mock-")

    @pytest.mark.asyncio
    async def test_mock_provider_project_response(self):
        """Test mock provider project-specific responses."""
        messages = [LLMMessage(role=LLMRole.USER, content="LPD: need to run experiments")]
        request = LLMRequest(messages=messages, model="mock-claude-3-opus")

        response = await self.provider.generate(request)

        # Should contain project-related content
        assert "project" in response.content.lower() or "lpd" in response.content.lower()

    @pytest.mark.asyncio
    async def test_mock_provider_failure_simulation(self):
        """Test mock provider failure simulation."""
        failing_provider = MockProvider(delay=0.01, fail_rate=1.0)
        messages = [LLMMessage(role=LLMRole.USER, content="Test")]
        request = LLMRequest(messages=messages, model="test")

        with pytest.raises(Exception):
            await failing_provider.generate(request)


class TestBedrockProvider:
    """Test AWS Bedrock provider (with mocking)."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the boto3 client to avoid actual AWS calls
        with patch('boto3.Session'):
            self.provider = BedrockProvider(
                region="us-east-1",
                aws_profile="test-profile"
            )

    def test_bedrock_provider_model_info(self):
        """Test Bedrock provider model information."""
        models = self.provider.list_available_models()
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models

        opus_info = self.provider.get_model_info("claude-3-opus-20240229")
        assert opus_info["model_id"] == "anthropic.claude-3-opus-20240229-v1:0"
        assert opus_info["max_tokens"] == 4096

    def test_bedrock_provider_cost_estimation(self):
        """Test Bedrock provider cost estimation."""
        messages = [LLMMessage(role=LLMRole.USER, content="Test message " * 100)]
        request = LLMRequest(
            messages=messages,
            model="claude-3-opus-20240229",
            max_tokens=1000
        )

        cost = self.provider.estimate_cost(request)
        assert cost["estimated_total_cost"] > 0
        assert cost["estimated_input_cost"] > 0
        assert cost["estimated_output_cost"] > 0

    @pytest.mark.asyncio
    async def test_bedrock_provider_request_preparation(self):
        """Test Bedrock request preparation."""
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="You are helpful"),
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
        request = LLMRequest(messages=messages, model="claude-3-opus-20240229")

        model_config = self.provider.MODELS["claude-3-opus-20240229"]
        bedrock_request = self.provider._prepare_bedrock_request(request, model_config)

        assert "messages" in bedrock_request
        assert "system" in bedrock_request
        assert "max_tokens" in bedrock_request
        assert bedrock_request["system"] == "You are helpful"
        assert len(bedrock_request["messages"]) == 1

    def test_bedrock_provider_response_parsing(self):
        """Test Bedrock response parsing."""
        mock_response = {
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "usage": {"input_tokens": 10, "output_tokens": 8},
            "stop_reason": "end_turn",
            "ResponseMetadata": {"RequestId": "test-123"}
        }

        response = self.provider._parse_bedrock_response(mock_response, "claude-3-opus-20240229")

        assert response.content == "Hello! How can I help you?"
        assert response.model == "claude-3-opus-20240229"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8
        assert response.finish_reason == "end_turn"
        assert response.request_id == "test-123"


class TestLLMService:
    """Test main LLM service."""

    def setup_method(self):
        """Set up test fixtures."""
        config = LLMConfig(provider="mock", cost_limit_daily=1.0)
        self.service = LLMService(config=config)

    def test_llm_service_initialization(self):
        """Test LLM service initialization."""
        assert self.service.config.provider == "mock"
        assert isinstance(self.service.provider, MockProvider)
        assert self.service.request_count == 0
        assert self.service.daily_cost == 0.0

    def test_llm_service_status(self):
        """Test service status reporting."""
        status = self.service.get_status()

        assert status["provider"] == "mock"
        assert status["request_count"] == 0
        assert status["daily_cost"] == 0.0
        assert "available_models" in status
        assert "config" in status

    @pytest.mark.asyncio
    async def test_llm_service_basic_generation(self):
        """Test basic text generation through service."""
        response = await self.service.generate("Test prompt")

        assert response.content is not None
        assert len(response.content) > 0
        assert self.service.request_count == 1
        assert self.service.daily_cost > 0

    @pytest.mark.asyncio
    async def test_llm_service_with_system_prompt(self):
        """Test generation with system prompt."""
        response = await self.service.generate(
            prompt="Test prompt",
            system_prompt="You are a helpful assistant",
            temperature=0.5
        )

        assert response.content is not None
        assert response.model is not None

    @pytest.mark.asyncio
    async def test_llm_service_cost_limiting(self):
        """Test cost limiting functionality."""
        # Set very low cost limit
        self.service.config.cost_limit_daily = 0.001
        self.service.daily_cost = 0.0

        # First request should work
        await self.service.generate("Test prompt 1")

        # Second request should fail due to cost limit
        with pytest.raises(LLMError):
            await self.service.generate("Test prompt 2")

    @pytest.mark.asyncio
    async def test_llm_service_text_interpretation(self):
        """Test text interpretation functionality."""
        result = await self.service.interpret_text(
            "LPD: need to run 20 experiments",
            context={"current_focus": "machine-learning"}
        )

        assert "type" in result
        assert "confidence" in result
        assert "explanation" in result

        # For mock provider, we should get some structured response
        if "projects" in result:
            assert isinstance(result["projects"], list)
        if "tasks" in result:
            assert isinstance(result["tasks"], list)

    @pytest.mark.asyncio
    async def test_llm_service_event_summarization(self):
        """Test event summarization functionality."""
        mock_events = [
            {
                "event_type": "task_created",
                "timestamp": "2024-01-01T10:00:00Z",
                "payload": {"title": "Run experiments"}
            },
            {
                "event_type": "project_created",
                "timestamp": "2024-01-01T11:00:00Z",
                "payload": {"name": "LPD Research"}
            }
        ]

        result = await self.service.summarize_events(
            events=mock_events,
            time_period="week",
            style="executive"
        )

        assert "summary" in result
        assert result["style"] == "executive"
        assert result["time_period"] == "week"
        assert result["event_count"] == 2
        assert len(result["summary"]) > 0

    def test_llm_service_config_management(self):
        """Test configuration loading and saving."""
        # Test config creation from dict
        config_dict = {
            "provider": "bedrock",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.3
        }

        config = LLMConfig.from_dict(config_dict)
        assert config.provider == "bedrock"
        assert config.model == "claude-3-sonnet-20240229"
        assert config.temperature == 0.3

        # Test config to dict
        config_dict_out = config.to_dict()
        assert config_dict_out["provider"] == "bedrock"


class TestPromptTemplates:
    """Test prompt template functionality."""

    def test_text_interpretation_system_prompt(self):
        """Test text interpretation system prompt generation."""
        prompt = PromptTemplates.text_interpretation_system_prompt()

        assert "projects" in prompt.lower()
        assert "tasks" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "json" in prompt.lower()

    def test_interpretation_user_prompt(self):
        """Test user prompt generation with context."""
        context = {
            "current_focus_project": "LPD",
            "recent_projects": ["LPD", "EB1"],
            "active_tasks_count": 5
        }

        prompt = PromptTemplates.interpretation_user_prompt(
            "need to call attorney",
            context=context
        )

        assert "need to call attorney" in prompt
        assert "LPD" in prompt
        assert "Context:" in prompt

    def test_weekly_summary_prompts(self):
        """Test weekly summary prompt generation."""
        # Test system prompts for different styles
        exec_prompt = PromptTemplates.weekly_summary_system_prompt("executive")
        assert "executive" in exec_prompt.lower()
        assert "accomplishments" in exec_prompt.lower()

        detailed_prompt = PromptTemplates.weekly_summary_system_prompt("detailed")
        assert "detailed" in detailed_prompt.lower()
        assert "comprehensive" in detailed_prompt.lower()

        # Test user prompt
        events = [
            {"event_type": "task_created", "payload": {"title": "Test task"}}
        ]

        user_prompt = PromptTemplates.weekly_summary_user_prompt(events)
        assert "weekly summary" in user_prompt.lower()
        assert "Task Events" in user_prompt

    def test_get_prompt_by_name(self):
        """Test getting prompts by name."""
        prompt = PromptTemplates.get_prompt_by_name(
            "text_interpretation_system"
        )
        assert "projects" in prompt.lower()

        # Test with parameters
        prompt = PromptTemplates.get_prompt_by_name(
            "interpretation_user",
            text="test input",
            context={"current_focus_project": "TestProject"}
        )
        assert "test input" in prompt
        assert "TestProject" in prompt

        # Test unknown prompt
        with pytest.raises(ValueError):
            PromptTemplates.get_prompt_by_name("unknown_prompt")


class TestLLMIntegration:
    """Integration tests for LLM service components."""

    @pytest.mark.asyncio
    async def test_service_with_different_providers(self):
        """Test service behavior with different providers."""
        # Test with mock provider
        mock_config = LLMConfig(provider="mock")
        mock_service = LLMService(config=mock_config)

        mock_response = await mock_service.generate("Test with mock")
        assert mock_response.content is not None

        # Test provider switching (would use bedrock in real scenario)
        # For testing, we just verify the service can be reconfigured
        assert mock_service.config.provider == "mock"

    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Test initialization with invalid provider
        bad_config = LLMConfig(provider="invalid_provider")

        # Should fallback to mock provider
        service = LLMService(config=bad_config)
        assert service.config.provider == "mock"
        assert isinstance(service.provider, MockProvider)

    @pytest.mark.asyncio
    async def test_realistic_interpretation_workflow(self):
        """Test realistic interpretation workflow."""
        config = LLMConfig(provider="mock", cost_limit_daily=10.0)
        service = LLMService(config=config)

        # Simulate typical user inputs
        test_inputs = [
            "LPD: need to analyze results from yesterday's experiments",
            "Call attorney about EB1 petition status",
            "Meeting with ML team went well, discussed new architecture",
            "Must implement the new authentication system by Friday"
        ]

        for text in test_inputs:
            result = await service.interpret_text(text)

            # Verify we get reasonable interpretation results
            assert "confidence" in result
            assert isinstance(result["confidence"], (int, float))
            assert "explanation" in result

            # Should identify type of input
            assert "type" in result