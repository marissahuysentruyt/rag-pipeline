"""
Document chunking strategies.

This package provides interfaces and implementations for various
chunking strategies (fixed-size, semantic, markdown-aware, etc.).
"""

from .base import (
    ChunkerStrategy,
    ChunkingConfig,
    Chunk,
    ChunkType,
    ChunkingError,
    InvalidChunkSizeError
)

__all__ = [
    'ChunkerStrategy',
    'ChunkingConfig',
    'Chunk',
    'ChunkType',
    'ChunkingError',
    'InvalidChunkSizeError'
]
