"""
Ingestion adapters for various document sources.

This package provides adapters for ingesting documents from different sources:
- Web crawlers
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

__all__ = [
    'IngestionAdapter',
    'Document',
    'DocumentMetadata',
    'IngestionError',
    'DocumentNotFoundError',
    'FetchError',
    'ConfigurationError'
]
