"""
Document retrieval strategies.

This package provides interfaces and implementations for various
retrieval strategies (vector similarity, hybrid, BM25, reranking, etc.).
"""

from .base import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievedDocument,
    RetrievalMethod,
    RetrievalResult,
    RetrievalError,
    QueryError,
    IndexNotFoundError
)

from .chroma_retriever import ChromaRetriever

__all__ = [
    'RetrievalStrategy',
    'RetrievalConfig',
    'RetrievedDocument',
    'RetrievalMethod',
    'RetrievalResult',
    'RetrievalError',
    'QueryError',
    'IndexNotFoundError',
    'ChromaRetriever'
]
