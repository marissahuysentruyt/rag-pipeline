"""
Chroma-based vector similarity retrieval strategy.

This module provides a concrete implementation of the RetrievalStrategy
interface using Chroma as the vector database.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from .base import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievedDocument,
    RetrievalResult,
    RetrievalMethod,
    RetrievalError
)
from src.embedding.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ChromaRetriever(RetrievalStrategy):
    """
    Vector similarity retrieval using Chroma database.

    This retriever uses embeddings to find semantically similar documents
    from a Chroma vector store.

    Example:
        >>> from src.embedding.providers import SentenceTransformersProvider
        >>> from src.embedding.providers import EmbeddingConfig
        >>>
        >>> # Create embedding provider
        >>> embedding_config = EmbeddingConfig(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     dimensions=384
        ... )
        >>> embedding_provider = SentenceTransformersProvider(embedding_config)
        >>> embedding_provider.load_model()
        >>>
        >>> # Create retriever
        >>> config = RetrievalConfig(top_k=5, score_threshold=0.3)
        >>> retriever = ChromaRetriever(
        ...     config=config,
        ...     document_store=document_store,
        ...     embedding_provider=embedding_provider
        ... )
        >>>
        >>> # Retrieve documents
        >>> result = retriever.retrieve("How do I use buttons?")
        >>> for doc in result.documents:
        ...     print(f"Score: {doc.score:.3f} - {doc.content[:100]}")
    """

    def __init__(
        self,
        config: RetrievalConfig,
        document_store: ChromaDocumentStore,
        embedding_provider: EmbeddingProvider
    ):
        """
        Initialize the Chroma retriever.

        Args:
            config: Retrieval configuration
            document_store: Chroma document store instance
            embedding_provider: Provider for generating query embeddings
        """
        super().__init__(config)
        self.document_store = document_store
        self.embedding_provider = embedding_provider

        # Verify document store has documents
        doc_count = self.document_store.count_documents()
        if doc_count == 0:
            logger.warning("Document store is empty. No documents to retrieve.")
        else:
            logger.info(f"Retriever initialized with {doc_count} documents")

    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.config.score_threshold is not None:
            if self.config.score_threshold < 0 or self.config.score_threshold > 1:
                raise ValueError("score_threshold must be between 0 and 1")

        if self.config.diversity_factor < 0 or self.config.diversity_factor > 1:
            raise ValueError("diversity_factor must be between 0 and 1")

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve documents relevant to the query using vector similarity.

        Args:
            query: User query
            filters: Optional metadata filters (e.g., {"domain": "example.com"})
            top_k: Override default top_k

        Returns:
            RetrievalResult with retrieved documents

        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

        # Use override or default top_k
        k = top_k if top_k is not None else self.config.top_k

        # Merge filters
        final_filters = self.config.filters.copy() if self.config.filters else {}
        if filters:
            final_filters.update(filters)

        try:
            logger.debug(f"Retrieving documents for query: '{query[:50]}...'")
            logger.debug(f"Filters: {final_filters}, top_k: {k}")

            # Generate query embedding
            query_embedding = self.embedding_provider.embed_text(query)

            # Search in Chroma
            # Note: Chroma's filter_documents doesn't directly support embedding search
            # We need to use the raw client or convert to Haystack retriever pattern
            results = self._search_with_embedding(
                query_embedding=query_embedding,
                top_k=k,
                filters=final_filters
            )

            # Convert to RetrievedDocument format
            retrieved_docs = []
            for idx, doc in enumerate(results):
                # Calculate score (Chroma returns distance, we convert to similarity)
                # Distance is typically L2 or cosine distance
                score = self._distance_to_score(doc.score) if hasattr(doc, 'score') else 0.5

                retrieved_docs.append(RetrievedDocument(
                    content=doc.content,
                    metadata=doc.meta if hasattr(doc, 'meta') else {},
                    score=score,
                    rank=idx + 1,
                    retrieval_method=RetrievalMethod.VECTOR_SIMILARITY,
                    chunk_id=doc.id if hasattr(doc, 'id') else None
                ))

            # Apply score filtering
            if self.config.score_threshold is not None:
                original_count = len(retrieved_docs)
                retrieved_docs = [
                    doc for doc in retrieved_docs
                    if doc.score >= self.config.score_threshold
                ]
                logger.debug(f"Filtered {original_count - len(retrieved_docs)} docs below threshold")

            # Apply diversity if configured
            if self.config.diversity_factor > 0:
                retrieved_docs = self._promote_diversity(retrieved_docs)

            # Apply reranking if configured
            if self.config.rerank:
                retrieved_docs = self._rerank_documents(query, retrieved_docs)

            total_candidates = self.document_store.count_documents()

            result = RetrievalResult(
                documents=retrieved_docs,
                query=query,
                retrieval_method=RetrievalMethod.VECTOR_SIMILARITY,
                total_candidates=total_candidates,
                filtered_count=len(retrieved_docs),
                metadata={
                    "top_k": k,
                    "filters": final_filters,
                    "score_threshold": self.config.score_threshold
                }
            )

            logger.info(
                f"Retrieved {len(retrieved_docs)} documents "
                f"(candidates: {total_candidates})"
            )

            return result

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}") from e

    def _search_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Search Chroma with embedding vector.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of documents
        """
        # Use Chroma's internal client for embedding search
        try:
            # Get Chroma collection
            collection = self.document_store._collection

            # Build where clause from filters
            where = None
            if filters:
                # Convert filters to Chroma's where format
                # Chroma uses {'field': value} or {'field': {'$eq': value}}
                where = {
                    k: v if not isinstance(v, (list, tuple)) else {"$in": v}
                    for k, v in filters.items()
                }

            # Query with embedding
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where
            )

            # Convert to document-like objects
            documents = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    # Create a simple document object
                    doc = type('Document', (), {})()
                    doc.content = results['documents'][0][i]
                    doc.meta = results['metadatas'][0][i] if results['metadatas'] else {}
                    doc.id = results['ids'][0][i] if results['ids'] else None
                    # Chroma returns distances, convert to scores
                    doc.score = results['distances'][0][i] if results['distances'] else None
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Chroma search failed: {str(e)}")
            raise

    def _distance_to_score(self, distance: float) -> float:
        """
        Convert distance to similarity score (0-1, higher is better).

        Chroma typically returns L2 or cosine distance.
        For cosine distance: similarity = 1 - distance
        For L2 distance: we normalize to 0-1 range

        Args:
            distance: Distance from Chroma

        Returns:
            Similarity score (0-1)
        """
        if distance is None:
            return 0.5

        # Assume cosine distance (0-2 range)
        # Convert to similarity (0-1 range, higher is better)
        score = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, score))

    def compute_similarity(self, query: str, document: str) -> float:
        """
        Compute similarity score between query and document.

        Args:
            query: Query text
            document: Document text

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Generate embeddings
            query_emb = self.embedding_provider.embed_text(query)
            doc_emb = self.embedding_provider.embed_text(document)

            # Compute cosine similarity
            similarity = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )

            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2

        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0

    def _promote_diversity(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Promote diversity in results using maximal marginal relevance.

        Args:
            documents: Retrieved documents

        Returns:
            Reordered documents with more diversity
        """
        if len(documents) <= 1:
            return documents

        # Simple diversity: penalize documents with similar content
        diverse_docs = [documents[0]]  # Start with top document

        for doc in documents[1:]:
            # Compute similarity to already selected documents
            max_similarity = 0.0
            for selected_doc in diverse_docs:
                similarity = self._compute_text_similarity(doc.content, selected_doc.content)
                max_similarity = max(max_similarity, similarity)

            # Adjust score based on diversity factor
            diversity_penalty = max_similarity * self.config.diversity_factor
            doc.score = doc.score * (1 - diversity_penalty)

            diverse_docs.append(doc)

        # Re-sort by adjusted scores
        diverse_docs.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for idx, doc in enumerate(diverse_docs):
            doc.rank = idx + 1

        return diverse_docs

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple text similarity (Jaccard similarity on words).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using a more sophisticated method.

        This is a placeholder for future reranking strategies
        (e.g., cross-encoder models).

        Args:
            query: Query text
            documents: Documents to rerank

        Returns:
            Reranked documents
        """
        logger.debug("Reranking not yet implemented, returning original order")
        return documents

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.

        Returns:
            Dictionary with stats
        """
        return {
            "total_documents": self.document_store.count_documents(),
            "top_k": self.config.top_k,
            "score_threshold": self.config.score_threshold,
            "retrieval_method": RetrievalMethod.VECTOR_SIMILARITY.value,
            "diversity_factor": self.config.diversity_factor,
            "rerank_enabled": self.config.rerank
        }


class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass
