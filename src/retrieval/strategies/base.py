"""
Base interface for retrieval strategies.

This module defines the abstract base class for retrieval strategies,
allowing the system to support multiple approaches to document retrieval
(vector similarity, hybrid, BM25, reranking, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class RetrievalMethod(Enum):
    """Methods for retrieving documents."""
    VECTOR_SIMILARITY = "vector_similarity"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


@dataclass
class RetrievedDocument:
    """
    A document retrieved from the index.

    Attributes:
        content: Document content
        metadata: Document metadata
        score: Relevance score (higher is better)
        rank: Rank in the results (1-indexed)
        retrieval_method: Method used to retrieve this document
        chunk_id: Unique identifier for the chunk
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    retrieval_method: RetrievalMethod
    chunk_id: Optional[str] = None


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval strategy.

    Attributes:
        top_k: Number of documents to retrieve
        score_threshold: Minimum score for a document to be included
        rerank: Whether to apply reranking
        diversity_factor: Factor for promoting diversity in results (0.0 to 1.0)
        filters: Metadata filters to apply
        additional_params: Strategy-specific parameters
    """
    top_k: int = 5
    score_threshold: Optional[float] = None
    rerank: bool = False
    diversity_factor: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """
    Result from document retrieval.

    Attributes:
        documents: List of retrieved documents
        query: Original query
        retrieval_method: Method used
        total_candidates: Total number of candidate documents
        filtered_count: Number after filtering
        metadata: Additional metadata about retrieval
    """
    documents: List[RetrievedDocument]
    query: str
    retrieval_method: RetrievalMethod
    total_candidates: int
    filtered_count: int
    metadata: Optional[Dict[str, Any]] = None


class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.

    Each strategy is responsible for:
    1. Querying the document index
    2. Computing relevance scores
    3. Filtering and ranking results
    4. Applying diversity constraints
    5. Optional reranking
    """

    def __init__(self, config: RetrievalConfig):
        """
        Initialize the retrieval strategy.

        Args:
            config: Retrieval configuration
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
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.

        Args:
            query: User query
            filters: Optional metadata filters
            top_k: Override default top_k

        Returns:
            RetrievalResult with retrieved documents

        Raises:
            RetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    def compute_similarity(
        self,
        query: str,
        document: str
    ) -> float:
        """
        Compute similarity score between query and document.

        Args:
            query: Query text
            document: Document text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        pass

    def filter_by_metadata(
        self,
        documents: List[RetrievedDocument],
        filters: Dict[str, Any]
    ) -> List[RetrievedDocument]:
        """
        Filter documents by metadata criteria.

        Args:
            documents: List of documents to filter
            filters: Metadata filters (e.g., {"domain": "example.com"})

        Returns:
            Filtered list of documents
        """
        filtered = []
        for doc in documents:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered

    def filter_by_score(
        self,
        documents: List[RetrievedDocument],
        threshold: float
    ) -> List[RetrievedDocument]:
        """
        Filter documents by minimum score.

        Args:
            documents: List of documents to filter
            threshold: Minimum score

        Returns:
            Documents with score >= threshold
        """
        return [doc for doc in documents if doc.score >= threshold]

    def rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Rerank documents for better relevance.

        Default implementation: no reranking (returns as-is).
        Override in subclasses to implement reranking logic.

        Args:
            query: User query
            documents: Documents to rerank

        Returns:
            Reranked documents
        """
        return documents

    def promote_diversity(
        self,
        documents: List[RetrievedDocument],
        factor: float
    ) -> List[RetrievedDocument]:
        """
        Promote diversity in results (avoid redundant documents).

        Args:
            documents: Documents to diversify
            factor: Diversity factor (0.0 = no diversity, 1.0 = max diversity)

        Returns:
            Diversified list of documents
        """
        if factor == 0.0 or len(documents) <= 1:
            return documents

        # Simple diversity: penalize documents with similar content
        # This is a basic implementation; override for better diversity
        diverse = [documents[0]]

        for doc in documents[1:]:
            # Check if too similar to already selected docs
            max_similarity = 0.0
            for selected in diverse:
                similarity = self._content_similarity(doc.content, selected.content)
                max_similarity = max(max_similarity, similarity)

            # Penalize score based on similarity
            adjusted_score = doc.score * (1 - factor * max_similarity)

            # Create new doc with adjusted score
            diverse_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=adjusted_score,
                rank=doc.rank,
                retrieval_method=doc.retrieval_method,
                chunk_id=doc.chunk_id
            )
            diverse.append(diverse_doc)

        # Re-sort by adjusted scores
        diverse.sort(key=lambda d: d.score, reverse=True)

        # Update ranks
        for i, doc in enumerate(diverse, start=1):
            doc.rank = i

        return diverse

    def _content_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple content similarity (Jaccard similarity on words).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the retrieval strategy.

        Returns:
            Dictionary with strategy metadata
        """
        return {
            "strategy": self.__class__.__name__,
            "retrieval_method": self.get_retrieval_method().value,
            "top_k": self.config.top_k,
            "score_threshold": self.config.score_threshold,
            "rerank": self.config.rerank,
            "diversity_factor": self.config.diversity_factor
        }

    @abstractmethod
    def get_retrieval_method(self) -> RetrievalMethod:
        """
        Get the retrieval method used by this strategy.

        Returns:
            RetrievalMethod enum value
        """
        pass


class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass


class QueryError(RetrievalError):
    """Raised when query processing fails."""
    pass


class IndexNotFoundError(RetrievalError):
    """Raised when document index is not found."""
    pass
