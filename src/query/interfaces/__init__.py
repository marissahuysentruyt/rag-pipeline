"""
Query interface implementations.

This package provides interfaces for different ways to query the RAG system
(CLI, REST API, Custom GPT, OpenAI-compatible API, etc.).
"""

from .base import (
    QueryInterface,
    QueryInterfaceType,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    QueryError,
    ValidationError,
    ProcessingError,
    FormattingError
)

__all__ = [
    'QueryInterface',
    'QueryInterfaceType',
    'QueryRequest',
    'QueryResponse',
    'SourceDocument',
    'QueryError',
    'ValidationError',
    'ProcessingError',
    'FormattingError'
]
