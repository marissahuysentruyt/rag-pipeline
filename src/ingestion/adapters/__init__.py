"""
Ingestion adapters for various document sources.

This package provides adapters for ingesting documents from different sources:
- Web crawlers
- Codebases (local source code)
- File systems
- CMS platforms (Contentful, Strapi, etc.)
- Databases (Postgres, MongoDB, etc.)
"""

from .base import (
    IngestionAdapter,
    Document,
    DocumentMetadata,
    IngestionError,
    DocumentNotFoundError,
    FetchError,
    ConfigurationError
)
from .codebase import CodebaseAdapter

__all__ = [
    'IngestionAdapter',
    'Document',
    'DocumentMetadata',
    'IngestionError',
    'DocumentNotFoundError',
    'FetchError',
    'ConfigurationError',
    'CodebaseAdapter',
]
