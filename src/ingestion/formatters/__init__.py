"""
Code entity formatters for converting parsed entities to documents.

This package provides formatters that convert CodeEntity objects
from parsers into Document objects suitable for indexing.
"""

from .code_entity_formatter import CodeEntityFormatter

__all__ = [
    'CodeEntityFormatter',
]
