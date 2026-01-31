"""
Ingestion module for RAG pipeline.

Provides document processing, file handling, and batch ingestion capabilities.
"""

from .document_processor import DocumentProcessor, DocumentChunk

__all__ = ["DocumentProcessor", "DocumentChunk"]
