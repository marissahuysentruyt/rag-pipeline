"""
Base adapter interface for document ingestion sources.

This module defines the abstract base class that all ingestion adapters
must implement, ensuring consistent behavior across different sources
(web crawlers, file systems, CMS, databases, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from pathlib import Path


@dataclass
class DocumentMetadata:
    """
    Metadata associated with an ingested document.

    Attributes:
        source_id: Unique identifier for the source (e.g., URL, file path, CMS ID)
        source_type: Type of source (e.g., "web", "file", "cms", "database")
        title: Document title
        url: Optional URL if document has a web location
        last_modified: When the document was last modified
        language: Document language code (e.g., "en", "es")
        additional_metadata: Any source-specific metadata
    """
    source_id: str
    source_type: str
    title: str
    url: Optional[str] = None
    last_modified: Optional[datetime] = None
    language: Optional[str] = "en"
    additional_metadata: Optional[Dict[str, Any]] = None


@dataclass
class Document:
    """
    Represents a document ingested from any source.

    Attributes:
        content: The raw text content of the document
        metadata: Associated metadata
        format: Content format (e.g., "markdown", "html", "plain_text", "json")
    """
    content: str
    metadata: DocumentMetadata
    format: str = "plain_text"


class IngestionAdapter(ABC):
    """
    Abstract base class for all ingestion adapters.

    Each adapter is responsible for:
    1. Connecting to a specific source type
    2. Discovering available documents
    3. Fetching document content
    4. Extracting and normalizing metadata
    5. Handling source-specific authentication and pagination
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with configuration.

        Args:
            config: Source-specific configuration (credentials, endpoints, etc.)
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate that required configuration parameters are present.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        pass

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the source.

        This might involve:
        - Authenticating with an API
        - Opening a database connection
        - Verifying file system access

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection and cleanup resources.
        """
        pass

    @abstractmethod
    def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List available document IDs from the source.

        Args:
            filters: Optional filters to narrow down documents
                    (e.g., {"type": "article", "status": "published"})
            limit: Maximum number of document IDs to return

        Returns:
            List of document IDs that can be fetched
        """
        pass

    @abstractmethod
    def fetch_document(self, document_id: str) -> Document:
        """
        Fetch a single document by its ID.

        Args:
            document_id: Unique identifier for the document

        Returns:
            Document object with content and metadata

        Raises:
            DocumentNotFoundError: If document doesn't exist
            FetchError: If fetching fails
        """
        pass

    @abstractmethod
    def fetch_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 10
    ) -> Iterator[Document]:
        """
        Fetch all documents from the source in batches.

        This is a generator that yields documents in batches to avoid
        loading everything into memory at once.

        Args:
            filters: Optional filters to narrow down documents
            batch_size: Number of documents to fetch per batch

        Yields:
            Document objects
        """
        pass

    @abstractmethod
    def get_updates_since(
        self,
        timestamp: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Iterator[Document]:
        """
        Fetch only documents that have been updated since a given timestamp.

        This enables incremental updates of the index.

        Args:
            timestamp: Only fetch documents modified after this time
            filters: Optional additional filters

        Yields:
            Updated Document objects
        """
        pass

    def supports_incremental_updates(self) -> bool:
        """
        Check if this adapter supports incremental updates.

        Returns:
            True if get_updates_since is supported, False otherwise
        """
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the source.

        Returns:
            Dictionary with stats like:
            {
                "total_documents": 1000,
                "last_updated": "2025-11-18T12:00:00",
                "source_type": "web"
            }
        """
        return {
            "source_type": self.__class__.__name__,
            "supports_incremental": self.supports_incremental_updates()
        }

    def __enter__(self):
        """Context manager entry: connect to source."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: disconnect from source."""
        self.disconnect()


class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class DocumentNotFoundError(IngestionError):
    """Raised when a requested document doesn't exist."""
    pass


class FetchError(IngestionError):
    """Raised when fetching a document fails."""
    pass


class ConfigurationError(IngestionError):
    """Raised when adapter configuration is invalid."""
    pass
