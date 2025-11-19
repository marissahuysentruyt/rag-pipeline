"""
Base interface for LLM providers.

This module defines the abstract base class for LLM providers,
allowing the system to support multiple language models from different
providers (Anthropic, OpenAI, Cohere, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from enum import Enum


class MessageRole(Enum):
    """Roles for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """
    A single message in a chat conversation.

    Attributes:
        role: Message role (system, user, or assistant)
        content: Message content
        name: Optional name for the message sender
        metadata: Additional metadata
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    """
    Configuration for LLM provider.

    Attributes:
        model_name: Name or identifier of the LLM
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: List of sequences that stop generation
        api_key: API key for authentication
        additional_params: Provider-specific parameters
    """
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    api_key: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """
    Result from text generation.

    Attributes:
        text: Generated text
        model: Model name used
        finish_reason: Why generation stopped (e.g., "stop", "length", "error")
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens generated
        total_tokens: Total tokens used
        metadata: Additional metadata from the provider
    """
    text: str
    model: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Each provider is responsible for:
    1. Authenticating with the LLM service
    2. Formatting prompts/messages correctly
    3. Generating text responses
    4. Handling streaming if supported
    5. Managing rate limits and retries
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            GenerationError: If generation fails
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> GenerationResult:
        """
        Generate a response in a chat conversation.

        Args:
            messages: List of chat messages (system, user, assistant)
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated response

        Raises:
            GenerationError: If generation fails
        """
        pass

    def generate_with_context(
        self,
        question: str,
        context: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate a response given a question and context (for RAG).

        Args:
            question: User question
            context: Context string or list of context chunks
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated answer
        """
        # Combine context if it's a list
        if isinstance(context, list):
            context_str = "\n\n".join(context)
        else:
            context_str = context

        # Build messages for chat format
        messages = []

        if system_prompt:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt
            ))

        # Add context and question as user message
        user_content = f"Context:\n{context_str}\n\nQuestion: {question}"
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=user_content
        ))

        return self.chat(messages, **kwargs)

    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming responses.

        Returns:
            True if streaming is supported
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming (async generator).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they're generated

        Raises:
            GenerationError: If generation fails
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text.

        Default implementation estimates based on characters.
        Override with provider-specific tokenizer if available.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "provider": self.__class__.__name__,
            "supports_streaming": self.supports_streaming()
        }

    def get_context_window(self) -> int:
        """
        Get the context window size for this model.

        Returns:
            Maximum number of tokens in context
        """
        # Default, should be overridden by specific providers
        return 4096

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources (close connections, etc.).
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: cleanup resources."""
        self.cleanup()


class GenerationError(Exception):
    """Base exception for generation errors."""
    pass


class AuthenticationError(GenerationError):
    """Raised when authentication fails."""
    pass


class RateLimitError(GenerationError):
    """Raised when rate limit is exceeded."""
    pass


class ContextLengthExceededError(GenerationError):
    """Raised when input exceeds context window."""
    pass
