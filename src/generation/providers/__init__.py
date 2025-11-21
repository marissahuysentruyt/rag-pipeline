"""
LLM provider implementations.

This package provides interfaces and implementations for various
LLM providers (Anthropic, OpenAI, Cohere, etc.).
"""

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

# Optional providers (may not be installed)
try:
    from .anthropic import AnthropicProvider
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AnthropicProvider = None

__all__ = [
    'LLMProvider',
    'LLMConfig',
    'ChatMessage',
    'MessageRole',
    'GenerationResult',
    'GenerationError',
    'AuthenticationError',
    'RateLimitError',
    'ContextLengthExceededError',
    'AnthropicProvider',
    'ANTHROPIC_AVAILABLE'
]
