"""
Embedding provider implementations.

This package provides interfaces and implementations for various
embedding providers (Sentence Transformers, OpenAI, Cohere, etc.).
"""

from .base import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingError,
    ModelLoadError,
    TokenLimitExceededError,
    RateLimitError
)

from .sentence_transformers import SentenceTransformersProvider

# Optional providers (may not be installed)
try:
    from .openai import OpenAIEmbeddingProvider
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIEmbeddingProvider = None

__all__ = [
    'EmbeddingProvider',
    'EmbeddingConfig',
    'EmbeddingResult',
    'EmbeddingError',
    'ModelLoadError',
    'TokenLimitExceededError',
    'RateLimitError',
    'SentenceTransformersProvider',
    'OpenAIEmbeddingProvider',
    'OPENAI_AVAILABLE'
]
