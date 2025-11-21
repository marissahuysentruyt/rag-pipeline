"""
Anthropic (Claude) LLM provider implementation.

This module provides a concrete implementation of the LLMProvider
interface using Anthropic's Claude models.
"""

import logging
from typing import List, Optional, Iterator
import os

try:
    from anthropic import Anthropic
    from anthropic import RateLimitError as AnthropicRateLimitError
    from anthropic import APIError, AuthenticationError as AnthropicAuthError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import (
    LLMProvider,
    LLMConfig,
    ChatMessage,
    MessageRole,
    GenerationResult,
    GenerationError,
    AuthenticationError,
    RateLimitError,
    ContextLengthExceededError
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    LLM provider using Anthropic's Claude models.

    Supports Claude 3 and later models:
    - claude-3-5-sonnet-20241022 (most capable, balanced)
    - claude-3-5-sonnet-20240620 (previous generation)
    - claude-3-opus-20240229 (most powerful)
    - claude-3-sonnet-20240229 (balanced)
    - claude-3-haiku-20240307 (fastest, most affordable)

    Example:
        >>> config = LLMConfig(
        ...     model_name="claude-sonnet-4-5-20250929",
        ...     temperature=0.7,
        ...     max_tokens=2048,
        ...     api_key="sk-ant-..."
        ... )
        >>> provider = AnthropicProvider(config)
        >>> result = provider.generate("What is RAG?")
        >>> print(result.text)
    """

    # Model context windows
    MODEL_CONTEXT_WINDOWS = {
        "claude-sonnet-4-5-20250929": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    def __init__(self, config: LLMConfig):
        """
        Initialize the Anthropic provider.

        Args:
            config: LLM configuration with api_key required
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        super().__init__(config)
        self._client: Optional[Anthropic] = None

    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.model_name:
            raise ValueError("model_name is required")

        if not self.config.api_key:
            raise ValueError("api_key is required for Anthropic provider")

        if self.config.temperature < 0 or self.config.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")

        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # Warn about unknown models
        if self.config.model_name not in self.MODEL_CONTEXT_WINDOWS:
            logger.warning(
                f"Unknown model {self.config.model_name}. "
                f"Known models: {list(self.MODEL_CONTEXT_WINDOWS.keys())}"
            )

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client if not already initialized."""
        if self._client is None:
            logger.info(f"Initializing Anthropic client for model: {self.config.model_name}")
            self._client = Anthropic(api_key=self.config.api_key)

    def _format_messages(self, messages: List[ChatMessage]) -> tuple[Optional[str], List[dict]]:
        """
        Format messages for Anthropic API.

        Anthropic requires system messages to be separate from the conversation.

        Args:
            messages: List of ChatMessage objects

        Returns:
            Tuple of (system_message, api_messages)
        """
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic uses a separate system parameter
                if system_message is None:
                    system_message = msg.content
                else:
                    # Concatenate multiple system messages
                    system_message += "\n\n" + msg.content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        return system_message, api_messages

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate text from a simple prompt.

        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters (override config)

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            GenerationError: If generation fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
        """
        # Convert prompt to messages
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[ChatMessage], **kwargs) -> GenerationResult:
        """
        Generate text from a conversation.

        Args:
            messages: List of ChatMessage objects
            **kwargs: Additional generation parameters (override config)

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            GenerationError: If generation fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            ContextLengthExceededError: If context is too long
        """
        self._initialize_client()

        try:
            # Format messages for Anthropic API
            system_message, api_messages = self._format_messages(messages)

            # Build API parameters
            api_params = {
                "model": self.config.model_name,
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }

            # Add system message if present
            if system_message:
                api_params["system"] = system_message

            # Add optional parameters
            if self.config.top_p != 1.0:
                api_params["top_p"] = self.config.top_p

            if self.config.top_k:
                api_params["top_k"] = self.config.top_k

            if self.config.stop_sequences:
                api_params["stop_sequences"] = self.config.stop_sequences

            # Call Anthropic API
            logger.debug(f"Calling Anthropic API with {len(api_messages)} messages")
            response = self._client.messages.create(**api_params)

            # Extract text from response
            text = response.content[0].text if response.content else ""

            # Build result
            result = GenerationResult(
                text=text,
                model=response.model,
                finish_reason=response.stop_reason,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                metadata={
                    "response_id": response.id,
                    "stop_reason": response.stop_reason,
                }
            )

            logger.info(
                f"Generated {result.completion_tokens} tokens "
                f"(total: {result.total_tokens})"
            )

            return result

        except AnthropicAuthError as e:
            raise AuthenticationError(
                f"Authentication failed: {str(e)}"
            ) from e
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                f"Rate limit exceeded: {str(e)}"
            ) from e
        except APIError as e:
            # Check for context length errors
            error_msg = str(e).lower()
            if "context" in error_msg and "length" in error_msg:
                raise ContextLengthExceededError(
                    f"Context too long: {str(e)}"
                ) from e
            raise GenerationError(
                f"API error: {str(e)}"
            ) from e
        except Exception as e:
            raise GenerationError(
                f"Failed to generate text: {str(e)}"
            ) from e

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with retrieved context (RAG pattern).

        Args:
            query: User query
            context: Retrieved context to include
            system_message: Optional system message
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text

        Example:
            >>> provider = AnthropicProvider(config)
            >>> result = provider.generate_with_context(
            ...     query="How do I use buttons?",
            ...     context="Button component docs...",
            ...     system_message="You are a design system expert."
            ... )
        """
        messages = []

        # Add system message if provided
        if system_message:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_message
            ))

        # Add user message with context
        user_content = f"""Context:
{context}

Question: {query}"""

        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=user_content
        ))

        return self.chat(messages, **kwargs)

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate text with streaming (yields tokens as they're generated).

        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks

        Raises:
            GenerationError: If generation fails
        """
        self._initialize_client()

        try:
            # Convert prompt to messages
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            system_message, api_messages = self._format_messages(messages)

            # Build API parameters
            api_params = {
                "model": self.config.model_name,
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": True,
            }

            if system_message:
                api_params["system"] = system_message

            # Stream response
            with self._client.messages.stream(**api_params) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise GenerationError(
                f"Failed to stream text: {str(e)}"
            ) from e

    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming.

        Returns:
            True (Anthropic supports streaming)
        """
        return True

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Approximate token count

        Note:
            This is a rough estimate. For accurate counts, use Anthropic's
            token counting API.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def get_model_info(self) -> dict:
        """
        Get information about the LLM model.

        Returns:
            Dictionary with model metadata
        """
        context_window = self.MODEL_CONTEXT_WINDOWS.get(
            self.config.model_name,
            200000  # Default for unknown models
        )

        return {
            "model_name": self.config.model_name,
            "provider": "Anthropic",
            "context_window": context_window,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_streaming": True,
            "supports_function_calling": True,
        }

    def cleanup(self) -> None:
        """Cleanup resources (close client connection)."""
        if self._client is not None:
            logger.info(f"Closing Anthropic client for {self.config.model_name}")
            # Anthropic client doesn't need explicit cleanup
            self._client = None

    def __enter__(self):
        """Context manager entry."""
        self._initialize_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: cleanup resources."""
        self.cleanup()
