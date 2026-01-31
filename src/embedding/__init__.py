"""
Embedding module for generating document embeddings.

This module provides interfaces and implementations for various
embedding providers.
"""

from .providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingError,
    ModelLoadError,
    TokenLimitExceededError,
    RateLimitError
)
from .factory import EmbeddingProviderFactory, EmbeddingProcessor

__all__ = [
    'EmbeddingProvider',
    'EmbeddingConfig',
    'EmbeddingResult',
    'EmbeddingError',
    'ModelLoadError',
    'TokenLimitExceededError',
    'RateLimitError',
    'EmbeddingProviderFactory',
    'EmbeddingProcessor'
]
