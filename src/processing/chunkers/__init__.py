"""
Document chunking strategies.

This package provides interfaces and implementations for various
chunking strategies (fixed-size, semantic, markdown-aware, code-aware, etc.).
"""

from .base import (
    ChunkerStrategy,
    ChunkingConfig,
    Chunk,
    ChunkType,
    ChunkingError,
    InvalidChunkSizeError
)
from .markdown import MarkdownChunker
from .code import CodeChunker

__all__ = [
    'ChunkerStrategy',
    'ChunkingConfig',
    'Chunk',
    'ChunkType',
    'ChunkingError',
    'InvalidChunkSizeError',
    'MarkdownChunker',
    'CodeChunker',
]
