"""
Base interface for query interfaces.

This module defines the protocol/interface for different ways to query
the RAG system (CLI, REST API, Custom GPT, OpenAI-compatible API, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator
from enum import Enum


class QueryInterfaceType(Enum):
    """Types of query interfaces."""
    CLI = "cli"
    REST_API = "rest_api"
    CUSTOM_GPT = "custom_gpt"
    OPENAI_API = "openai_api"
    GRAPHQL = "graphql"
    GRPC = "grpc"


@dataclass
class QueryRequest:
    """
    A query request from any interface.

    Attributes:
        question: The user's question
        filters: Optional metadata filters
        top_k: Number of documents to retrieve
        domain: Optional domain filter
        session_id: Optional session ID for conversation tracking
        user_id: Optional user ID for personalization
        additional_params: Interface-specific parameters
    """
    question: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    domain: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class SourceDocument:
    """
    A source document included in the response.

    Attributes:
        title: Document title
        url: Document URL
        content: Document content (may be truncated)
        score: Relevance score
        metadata: Additional metadata
    """
    title: str
    url: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResponse:
    """
    A query response to any interface.

    Attributes:
        answer: Generated answer text
        sources: Source documents used
        metadata: Response metadata (model, timing, etc.)
        session_id: Session ID if conversation tracking is enabled
        error: Error message if query failed
    """
    answer: str
    sources: List[SourceDocument]
    metadata: Dict[str, Any]
    session_id: Optional[str] = None
    error: Optional[str] = None


class QueryInterface(ABC):
    """
    Abstract base class for query interfaces.

    Each interface is responsible for:
    1. Receiving queries in its specific format
    2. Converting to standard QueryRequest
    3. Invoking the RAG pipeline
    4. Formatting response appropriately
    5. Handling errors gracefully
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the query interface.

        Args:
            config: Interface-specific configuration
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query request and return a response.

        Args:
            request: Standard query request

        Returns:
            QueryResponse with answer and sources

        Raises:
            QueryError: If query processing fails
        """
        pass

    @abstractmethod
    def format_response(
        self,
        response: QueryResponse,
        format_type: Optional[str] = None
    ) -> Any:
        """
        Format the response for this specific interface.

        Args:
            response: Standard query response
            format_type: Optional format specifier (e.g., "json", "markdown", "html")

        Returns:
            Formatted response (type depends on interface)
        """
        pass

    def validate_request(self, request: QueryRequest) -> bool:
        """
        Validate a query request.

        Args:
            request: Query request to validate

        Returns:
            True if valid, raises exception otherwise

        Raises:
            ValidationError: If request is invalid
        """
        if not request.question or not request.question.strip():
            raise ValidationError("Question cannot be empty")

        if request.top_k < 1 or request.top_k > 50:
            raise ValidationError("top_k must be between 1 and 50")

        return True

    def supports_streaming(self) -> bool:
        """
        Check if this interface supports streaming responses.

        Returns:
            True if streaming is supported
        """
        return False

    @abstractmethod
    async def process_query_stream(
        self,
        request: QueryRequest
    ) -> AsyncIterator[str]:
        """
        Process a query with streaming response.

        Args:
            request: Query request

        Yields:
            Response chunks as they're generated

        Raises:
            QueryError: If query processing fails
            NotImplementedError: If streaming is not supported
        """
        pass

    def get_interface_type(self) -> QueryInterfaceType:
        """
        Get the type of this interface.

        Returns:
            QueryInterfaceType enum value
        """
        return QueryInterfaceType.REST_API  # Default, override in subclasses

    def get_interface_info(self) -> Dict[str, Any]:
        """
        Get information about this interface.

        Returns:
            Dictionary with interface metadata
        """
        return {
            "interface_type": self.get_interface_type().value,
            "interface_class": self.__class__.__name__,
            "supports_streaming": self.supports_streaming(),
            "config": self.config
        }

    def handle_error(self, error: Exception, request: QueryRequest) -> QueryResponse:
        """
        Handle errors and return appropriate error response.

        Args:
            error: The exception that occurred
            request: The original request

        Returns:
            QueryResponse with error information
        """
        error_message = f"{error.__class__.__name__}: {str(error)}"

        return QueryResponse(
            answer="",
            sources=[],
            metadata={
                "error": True,
                "error_type": error.__class__.__name__,
                "request": {
                    "question": request.question,
                    "top_k": request.top_k
                }
            },
            error=error_message
        )


class QueryError(Exception):
    """Base exception for query errors."""
    pass


class ValidationError(QueryError):
    """Raised when request validation fails."""
    pass


class ProcessingError(QueryError):
    """Raised when query processing fails."""
    pass


class FormattingError(QueryError):
    """Raised when response formatting fails."""
    pass
