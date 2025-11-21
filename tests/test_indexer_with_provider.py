"""
Tests for DocumentIndexer using EmbeddingProvider interface.

This tests the new modular architecture where DocumentIndexer uses
the EmbeddingProvider interface instead of Haystack embedders.
"""

import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.ingestion.document_indexer import DocumentIndexer
from src.ingestion.document_processor import DocumentChunk
from src.embedding.providers.base import EmbeddingConfig
from src.embedding.providers.sentence_transformers import SentenceTransformersProvider


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary directory for test database."""
    db_path = tmp_path / "test_chroma_provider_db"
    db_path.mkdir()
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        shutil.rmtree(db_path)


@pytest.fixture
def embedding_config():
    """Create embedding configuration."""
    return EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        batch_size=16,
        normalize=True
    )


@pytest.fixture
def embedding_provider(embedding_config):
    """Create and load an embedding provider."""
    provider = SentenceTransformersProvider(embedding_config)
    provider.load_model()
    yield provider
    provider.cleanup()


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            content="Button component allows users to trigger actions.",
            metadata={
                "title": "Button",
                "url": "https://example.com/button",
                "domain": "example.com"
            },
            chunk_type="text",
            heading="Button Component"
        ),
        DocumentChunk(
            content="Input component is used for text entry.",
            metadata={
                "title": "Input",
                "url": "https://example.com/input",
                "domain": "example.com"
            },
            chunk_type="text",
            heading="Input Component"
        ),
    ]


class TestProviderMode:
    """Tests for DocumentIndexer using EmbeddingProvider."""

    def test_initialization_with_provider(self, test_db_path, embedding_provider):
        """Test initialization with EmbeddingProvider."""
        indexer = DocumentIndexer(
            collection_name="test_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        assert indexer.use_provider is True
        assert indexer.embedding_provider is embedding_provider
        assert indexer.embedder is None
        assert indexer.indexing_pipeline is None

    def test_initialization_without_provider(self, test_db_path):
        """Test initialization without EmbeddingProvider (legacy mode)."""
        indexer = DocumentIndexer(
            collection_name="test_legacy",
            persist_path=test_db_path
        )

        assert indexer.use_provider is False
        assert indexer.embedding_provider is None
        assert indexer.embedder is not None
        assert indexer.indexing_pipeline is not None

    def test_index_chunks_with_provider(self, test_db_path, embedding_provider, sample_chunks):
        """Test indexing chunks using EmbeddingProvider."""
        indexer = DocumentIndexer(
            collection_name="test_provider_index",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        count = indexer.index_chunks(sample_chunks, batch_size=10)

        assert count == 2
        assert indexer.document_store.count_documents() == 2

    def test_embeddings_generated_with_provider(self, test_db_path, embedding_provider, sample_chunks):
        """Test that embeddings are actually generated with provider."""
        indexer = DocumentIndexer(
            collection_name="test_embeddings_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        indexer.index_chunks(sample_chunks)

        # Retrieve documents to check embeddings
        docs = indexer.document_store.filter_documents()
        assert len(docs) == 2

        # Check that all documents have embeddings
        for doc in docs:
            assert doc.embedding is not None
            assert len(doc.embedding) == 384  # Dimension of all-MiniLM-L6-v2

    def test_batch_processing_with_provider(self, test_db_path, embedding_provider):
        """Test batch processing with EmbeddingProvider."""
        # Create many chunks to test batching
        chunks = [
            DocumentChunk(
                content=f"Test document {i}",
                metadata={"id": i},
                chunk_type="text",
                heading=f"Document {i}"
            )
            for i in range(25)
        ]

        indexer = DocumentIndexer(
            collection_name="test_batch_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        count = indexer.index_chunks(chunks, batch_size=10)

        assert count == 25
        assert indexer.document_store.count_documents() == 25

    def test_metadata_preserved_with_provider(self, test_db_path, embedding_provider):
        """Test that metadata is preserved when using provider."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={
                "title": "Test",
                "url": "https://test.com",
                "domain": "test.com",
                "custom_field": "custom_value"
            },
            chunk_type="code",
            heading="Test Section"
        )

        indexer = DocumentIndexer(
            collection_name="test_metadata_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        indexer.index_chunks([chunk])

        # Retrieve and check metadata
        docs = indexer.document_store.filter_documents()
        assert len(docs) == 1

        doc = docs[0]
        assert doc.meta["title"] == "Test"
        assert doc.meta["url"] == "https://test.com"
        assert doc.meta["domain"] == "test.com"
        assert doc.meta["custom_field"] == "custom_value"
        assert doc.meta["chunk_type"] == "code"
        assert doc.meta["heading"] == "Test Section"


class TestProviderVsLegacy:
    """Tests comparing provider mode vs legacy mode."""

    def test_both_modes_produce_documents(self, test_db_path, embedding_provider, sample_chunks):
        """Test that both modes successfully index documents."""
        # Index with provider
        indexer_provider = DocumentIndexer(
            collection_name="test_provider_mode",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )
        count_provider = indexer_provider.index_chunks(sample_chunks)

        # Create new path for legacy mode
        legacy_path = str(Path(test_db_path).parent / "legacy_db")
        Path(legacy_path).mkdir(exist_ok=True)

        # Index with legacy
        indexer_legacy = DocumentIndexer(
            collection_name="test_legacy_mode",
            persist_path=legacy_path
        )
        count_legacy = indexer_legacy.index_chunks(sample_chunks)

        assert count_provider == count_legacy == 2
        assert indexer_provider.document_store.count_documents() == 2
        assert indexer_legacy.document_store.count_documents() == 2

        # Cleanup legacy path
        if Path(legacy_path).exists():
            shutil.rmtree(legacy_path)

    def test_embedding_dimensions_match(self, test_db_path, embedding_provider, sample_chunks):
        """Test that embeddings from both modes have same dimensions."""
        # Index with provider
        indexer_provider = DocumentIndexer(
            collection_name="test_dim_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )
        indexer_provider.index_chunks([sample_chunks[0]])

        # Create new path for legacy mode
        legacy_path = str(Path(test_db_path).parent / "legacy_dim_db")
        Path(legacy_path).mkdir(exist_ok=True)

        # Index with legacy
        indexer_legacy = DocumentIndexer(
            collection_name="test_dim_legacy",
            persist_path=legacy_path
        )
        indexer_legacy.index_chunks([sample_chunks[0]])

        # Get embeddings
        docs_provider = indexer_provider.document_store.filter_documents()
        docs_legacy = indexer_legacy.document_store.filter_documents()

        assert len(docs_provider[0].embedding) == len(docs_legacy[0].embedding) == 384

        # Cleanup legacy path
        if Path(legacy_path).exists():
            shutil.rmtree(legacy_path)


class TestProviderIntegration:
    """Integration tests with real embedding provider."""

    def test_real_embeddings_with_provider(self, test_db_path, embedding_provider):
        """Test with real sentence transformers model."""
        chunks = [
            DocumentChunk(
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"topic": "ML"},
                chunk_type="text",
                heading="ML Overview"
            ),
            DocumentChunk(
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"topic": "DL"},
                chunk_type="text",
                heading="DL Overview"
            ),
        ]

        indexer = DocumentIndexer(
            collection_name="test_real_embeddings",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        indexer.index_chunks(chunks)

        # Verify documents were indexed
        docs = indexer.document_store.filter_documents()
        assert len(docs) == 2

        # Verify embeddings are different (content is different)
        import numpy as np
        embedding1 = np.array(docs[0].embedding)
        embedding2 = np.array(docs[1].embedding)

        # Embeddings should not be identical
        assert not np.allclose(embedding1, embedding2)

        # Embeddings should be normalized (unit length)
        assert np.isclose(np.linalg.norm(embedding1), 1.0, atol=0.01)
        assert np.isclose(np.linalg.norm(embedding2), 1.0, atol=0.01)

    def test_empty_chunks_handled_by_provider(self, test_db_path, embedding_provider):
        """Test that empty chunks are handled gracefully."""
        indexer = DocumentIndexer(
            collection_name="test_empty_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        count = indexer.index_chunks([])
        assert count == 0


class TestProviderStats:
    """Tests for statistics with provider mode."""

    def test_get_stats_with_provider(self, test_db_path, embedding_provider, sample_chunks):
        """Test getting stats when using provider."""
        indexer = DocumentIndexer(
            collection_name="test_stats_provider",
            persist_path=test_db_path,
            embedding_provider=embedding_provider
        )

        indexer.index_chunks(sample_chunks)
        stats = indexer.get_stats()

        assert stats["total_documents"] == 2
        assert stats["collection_name"] == "test_stats_provider"
        assert stats["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert test_db_path in stats["persist_path"]
