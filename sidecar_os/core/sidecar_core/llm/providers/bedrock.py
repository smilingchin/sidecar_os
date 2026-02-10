"""AWS Bedrock LLM provider implementation."""

import json
import asyncio
import boto3
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError, BotoCoreError
import logging

from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMMessage,
    LLMRole,
    LLMUsage,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMServiceError,
)

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider with support for Claude 3 models."""

    # Model configurations
    MODELS = {
        "claude-opus-4.6": {
            "model_id": "us.anthropic.claude-opus-4-6-v1",
            "max_tokens": 8192,
            "input_cost_per_1k": 0.015,  # Estimated - will need to check actual pricing
            "output_cost_per_1k": 0.075,
        },
        "claude-3-opus-20240229": {
            "model_id": "anthropic.claude-3-opus-20240229-v1:0",
            "max_tokens": 4096,
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
        },
        "claude-3-sonnet-20240229": {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 4096,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015,
        },
        "claude-3-haiku-20240307": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "max_tokens": 4096,
            "input_cost_per_1k": 0.00025,
            "output_cost_per_1k": 0.00125,
        },
    }

    def __init__(
        self,
        region: str = "us-east-1",
        aws_profile: Optional[str] = None,
        **boto3_kwargs
    ):
        """Initialize Bedrock provider.

        Args:
            region: AWS region for Bedrock
            aws_profile: AWS profile name to use
            **boto3_kwargs: Additional boto3 client arguments
        """
        self.region = region
        self.aws_profile = aws_profile

        # Create boto3 session with profile if specified
        session_kwargs = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile

        session = boto3.Session(**session_kwargs)

        try:
            self.client = session.client(
                "bedrock-runtime",
                region_name=region,
                **boto3_kwargs
            )
            logger.info(f"Initialized Bedrock client for region {region}")
        except Exception as e:
            raise LLMAuthenticationError(f"Failed to initialize Bedrock client: {e}")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using AWS Bedrock.

        Args:
            request: LLM request

        Returns:
            LLM response

        Raises:
            LLMError: Various errors during generation
        """
        self.validate_request(request)

        model_config = self.MODELS.get(request.model)
        if not model_config:
            raise ValueError(f"Unsupported model: {request.model}")

        # Prepare Bedrock request
        bedrock_request = self._prepare_bedrock_request(request, model_config)

        try:
            # Make async call to Bedrock
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._invoke_model,
                model_config["model_id"],
                bedrock_request
            )

            return self._parse_bedrock_response(response, request.model)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            if error_code == "ThrottlingException":
                raise LLMRateLimitError(f"Rate limit exceeded: {e}")
            elif error_code in ("AccessDeniedException", "UnauthorizedOperation"):
                raise LLMAuthenticationError(f"Authentication failed: {e}")
            elif error_code in ("ServiceUnavailableException", "InternalServerError"):
                raise LLMServiceError(f"Bedrock service error: {e}")
            else:
                raise LLMError(f"Bedrock API error ({error_code}): {e}")

        except BotoCoreError as e:
            raise LLMError(f"Boto3 error: {e}")

        except Exception as e:
            raise LLMError(f"Unexpected error during generation: {e}")

    def _prepare_bedrock_request(self, request: LLMRequest, model_config: Dict) -> Dict:
        """Prepare request for Bedrock API.

        Args:
            request: Original LLM request
            model_config: Model configuration

        Returns:
            Bedrock API request body
        """
        # Convert messages to Claude format
        messages = []
        system_message = None

        for msg in request.messages:
            if msg.role == LLMRole.SYSTEM:
                system_message = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        # Build Bedrock request body
        body = {
            "messages": messages,
            "max_tokens": min(
                request.max_tokens or model_config["max_tokens"],
                model_config["max_tokens"]
            ),
            "anthropic_version": "bedrock-2023-05-31",
        }

        # Add system message if present
        if system_message or request.system_prompt:
            body["system"] = system_message or request.system_prompt

        # Add optional parameters
        if request.temperature is not None:
            body["temperature"] = request.temperature

        if request.stop_sequences:
            body["stop_sequences"] = request.stop_sequences

        return body

    def _invoke_model(self, model_id: str, body: Dict) -> Dict:
        """Synchronous model invocation for executor.

        Args:
            model_id: Bedrock model ID
            body: Request body

        Returns:
            Bedrock response
        """
        logger.debug(f"Invoking Bedrock model {model_id}")

        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        return json.loads(response["body"].read())

    def _parse_bedrock_response(self, response: Dict, model: str) -> LLMResponse:
        """Parse Bedrock response into LLMResponse.

        Args:
            response: Raw Bedrock response
            model: Model name

        Returns:
            Parsed LLM response
        """
        # Extract content from Claude response
        content = ""
        if "content" in response:
            # Claude 3 format
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")
        else:
            # Fallback for other formats
            content = response.get("completion", "")

        # Extract usage information
        usage = None
        if "usage" in response:
            usage_data = response["usage"]
            usage = LLMUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
            )

        return LLMResponse(
            content=content.strip(),
            model=model,
            usage=usage,
            finish_reason=response.get("stop_reason"),
            request_id=response.get("ResponseMetadata", {}).get("RequestId"),
            metadata={
                "bedrock_response": response
            }
        )

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        return self.MODELS[model_name].copy()

    def list_available_models(self) -> List[str]:
        """List available models.

        Returns:
            List of model names
        """
        return list(self.MODELS.keys())

    def estimate_cost(self, request: LLMRequest) -> Dict[str, float]:
        """Estimate request cost.

        Args:
            request: LLM request

        Returns:
            Cost estimates
        """
        model_config = self.MODELS.get(request.model)
        if not model_config:
            return {"error": "Unknown model"}

        # Rough token estimation (4 chars per token)
        input_chars = sum(len(msg.content) for msg in request.messages)
        estimated_input_tokens = input_chars / 4

        # Estimate output tokens based on max_tokens or reasonable default
        estimated_output_tokens = request.max_tokens or 1000

        input_cost = (estimated_input_tokens / 1000) * model_config["input_cost_per_1k"]
        output_cost = (estimated_output_tokens / 1000) * model_config["output_cost_per_1k"]

        return {
            "estimated_input_tokens": int(estimated_input_tokens),
            "estimated_output_tokens": int(estimated_output_tokens),
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": input_cost + output_cost,
        }