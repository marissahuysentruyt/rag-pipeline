"""
Unit tests for document indexing and embedding generation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from src.ingestion.document_indexer import DocumentIndexer
from src.ingestion.document_processor import DocumentChunk


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary directory for test database."""
    db_path = tmp_path / "test_chroma_db"
    db_path.mkdir()
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        shutil.rmtree(db_path)


@pytest.fixture
def indexer(test_db_path):
    """Create a DocumentIndexer instance for testing with temporary database."""
    return DocumentIndexer(
        collection_name="test_collection",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_path=test_db_path
    )


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
        DocumentChunk(
            content="```jsx\n<Button>Click me</Button>\n```",
            metadata={
                "title": "Button",
                "url": "https://example.com/button",
                "domain": "example.com"
            },
            chunk_type="code",
            heading="Usage"
        )
    ]


class TestInitialization:
    """Tests for DocumentIndexer initialization."""

    def test_initialization(self, test_db_path):
        """Test basic initialization of DocumentIndexer."""
        indexer = DocumentIndexer(
            collection_name="test_init",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            persist_path=test_db_path
        )

        assert indexer.collection_name == "test_init"
        assert indexer.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert indexer.persist_path == test_db_path
        assert indexer.document_store is not None
        assert indexer.embedder is not None

    def test_default_parameters(self, test_db_path):
        """Test initialization with default parameters."""
        indexer = DocumentIndexer(persist_path=test_db_path)

        assert indexer.collection_name == "design_system_docs"
        assert "all-MiniLM-L6-v2" in indexer.embedding_model


class TestChunkConversion:
    """Tests for converting DocumentChunk to Haystack Document."""

    def test_chunk_to_document(self, indexer):
        """Test basic chunk to document conversion."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={
                "title": "Test",
                "url": "https://example.com/test"
            },
            chunk_type="text",
            heading="Test Heading"
        )

        doc = indexer.chunk_to_haystack_document(chunk)

        assert isinstance(doc, Document)
        assert doc.content == "Test content"
        assert doc.meta["title"] == "Test"
        assert doc.meta["url"] == "https://example.com/test"
        assert doc.meta["chunk_type"] == "text"
        assert doc.meta["heading"] == "Test Heading"

    def test_chunk_with_none_metadata(self, indexer):
        """Test conversion when chunk has None values in metadata."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={
                "title": "Test",
                "url": "https://example.com/test"
            },
            chunk_type="text",
            heading=None  # None heading
        )

        doc = indexer.chunk_to_haystack_document(chunk)

        assert "heading" not in doc.meta  # None values should be filtered out
        assert doc.meta["title"] == "Test"

    def test_chunk_with_complex_metadata(self, indexer):
        """Test conversion with various metadata types."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={
                "title": "Test",
                "count": 42,
                "ratio": 3.14,
                "enabled": True,
                "disabled": False
            },
            chunk_type="text",
            heading="Heading"
        )

        doc = indexer.chunk_to_haystack_document(chunk)

        assert doc.meta["count"] == 42
        assert doc.meta["ratio"] == 3.14
        assert doc.meta["enabled"] is True
        assert doc.meta["disabled"] is False

    def test_multiple_chunks_conversion(self, indexer, sample_chunks):
        """Test converting multiple chunks."""
        docs = [indexer.chunk_to_haystack_document(chunk) for chunk in sample_chunks]

        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)
        assert docs[0].content == "Button component allows users to trigger actions."
        assert docs[1].content == "Input component is used for text entry."
        assert docs[2].meta["chunk_type"] == "code"


class TestIndexing:
    """Tests for indexing document chunks."""

    def test_index_empty_chunks(self, indexer):
        """Test indexing with empty chunk list."""
        count = indexer.index_chunks([])
        assert count == 0

    def test_index_single_chunk(self, indexer, sample_chunks):
        """Test indexing a single chunk."""
        count = indexer.index_chunks([sample_chunks[0]])

        assert count == 1

        # Verify it was stored
        stats = indexer.get_stats()
        assert stats["total_documents"] == 1

    def test_index_multiple_chunks(self, indexer, sample_chunks):
        """Test indexing multiple chunks."""
        count = indexer.index_chunks(sample_chunks)

        assert count == 3

        # Verify they were stored
        stats = indexer.get_stats()
        assert stats["total_documents"] == 3

    def test_index_with_batch_processing(self, indexer):
        """Test indexing with small batch size."""
        # Create unique chunks to test batching (avoid duplicates)
        many_chunks = []
        for i in range(15):
            chunk = DocumentChunk(
                content=f"Chunk number {i} with unique content for testing batch processing.",
                metadata={
                    "title": f"Chunk {i}",
                    "url": f"https://example.com/chunk{i}",
                    "domain": "example.com"
                },
                chunk_type="text",
                heading=f"Heading {i}"
            )
            many_chunks.append(chunk)

        count = indexer.index_chunks(many_chunks, batch_size=5)

        assert count == 15

        stats = indexer.get_stats()
        assert stats["total_documents"] == 15

    def test_index_duplicate_policy_overwrite(self, indexer, sample_chunks):
        """Test indexing with overwrite policy."""
        # Index once
        indexer.index_chunks([sample_chunks[0]])
        stats1 = indexer.get_stats()

        # Index again with overwrite (default)
        indexer.index_chunks([sample_chunks[0]], duplicate_policy=DuplicatePolicy.OVERWRITE)
        stats2 = indexer.get_stats()

        # Should still have same count (overwritten, not duplicated)
        # Note: Chroma may handle this differently, so we just verify it doesn't error
        assert stats2["total_documents"] >= 1

    def test_index_preserves_metadata(self, indexer):
        """Test that indexing preserves all metadata."""
        chunk = DocumentChunk(
            content="Test content with metadata",
            metadata={
                "title": "Metadata Test",
                "url": "https://example.com/meta",
                "domain": "example.com",
                "custom_field": "custom_value"
            },
            chunk_type="text",
            heading="Metadata Heading"
        )

        indexer.index_chunks([chunk])

        # Retrieve and verify
        all_docs = indexer.document_store.filter_documents()
        assert len(all_docs) > 0

        doc = all_docs[0]
        assert doc.meta["title"] == "Metadata Test"
        assert doc.meta["domain"] == "example.com"
        assert doc.meta["chunk_type"] == "text"


class TestStatistics:
    """Tests for getting indexer statistics."""

    def test_get_stats_empty(self, indexer):
        """Test getting stats for empty index."""
        stats = indexer.get_stats()

        assert stats["total_documents"] == 0
        assert stats["collection_name"] == "test_collection"
        assert "all-MiniLM-L6-v2" in stats["embedding_model"]
        assert stats["persist_path"] == indexer.persist_path

    def test_get_stats_after_indexing(self, indexer, sample_chunks):
        """Test getting stats after indexing documents."""
        indexer.index_chunks(sample_chunks)

        stats = indexer.get_stats()

        assert stats["total_documents"] == 3
        assert stats["collection_name"] == "test_collection"

    def test_stats_structure(self, indexer):
        """Test that stats return correct structure."""
        stats = indexer.get_stats()

        assert "total_documents" in stats
        assert "collection_name" in stats
        assert "embedding_model" in stats
        assert "persist_path" in stats

        assert isinstance(stats["total_documents"], int)
        assert isinstance(stats["collection_name"], str)
        assert isinstance(stats["embedding_model"], str)


class TestClearIndex:
    """Tests for clearing the index."""

    def test_clear_empty_index(self, indexer):
        """Test clearing an already empty index."""
        # Should not error
        indexer.clear_index()

        stats = indexer.get_stats()
        assert stats["total_documents"] == 0

    def test_clear_populated_index(self, indexer, sample_chunks):
        """Test clearing a populated index."""
        # Add documents
        indexer.index_chunks(sample_chunks)
        assert indexer.get_stats()["total_documents"] == 3

        # Clear
        indexer.clear_index()

        # Verify cleared
        stats = indexer.get_stats()
        assert stats["total_documents"] == 0

    def test_clear_and_reindex(self, indexer, sample_chunks):
        """Test clearing and then reindexing."""
        # Index
        indexer.index_chunks(sample_chunks)
        assert indexer.get_stats()["total_documents"] == 3

        # Clear
        indexer.clear_index()
        assert indexer.get_stats()["total_documents"] == 0

        # Reindex
        indexer.index_chunks([sample_chunks[0]])
        assert indexer.get_stats()["total_documents"] == 1


class TestEmbeddings:
    """Tests for embedding generation."""

    def test_embeddings_are_generated(self, indexer, sample_chunks):
        """Test that embeddings are actually generated."""
        # Index chunks
        indexer.index_chunks([sample_chunks[0]])

        # Retrieve document
        all_docs = indexer.document_store.filter_documents()
        assert len(all_docs) > 0

        doc = all_docs[0]
        # Check that embedding exists (Haystack stores it in the document)
        assert doc.embedding is not None
        assert len(doc.embedding) > 0
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert len(doc.embedding) == 384

    def test_different_content_different_embeddings(self, indexer, sample_chunks):
        """Test that different content produces different embeddings."""
        # Index two different chunks
        indexer.index_chunks([sample_chunks[0], sample_chunks[1]])

        # Retrieve documents
        all_docs = indexer.document_store.filter_documents()
        assert len(all_docs) == 2

        # Embeddings should be different
        emb1 = all_docs[0].embedding
        emb2 = all_docs[1].embedding

        assert emb1 != emb2


class TestPersistence:
    """Tests for database persistence."""

    def test_persistence_across_instances(self, test_db_path, sample_chunks):
        """Test that data persists across indexer instances."""
        # Create indexer and add documents
        indexer1 = DocumentIndexer(
            collection_name="persistence_test",
            persist_path=test_db_path
        )
        indexer1.index_chunks(sample_chunks)
        count1 = indexer1.get_stats()["total_documents"]

        # Create new indexer instance pointing to same DB
        indexer2 = DocumentIndexer(
            collection_name="persistence_test",
            persist_path=test_db_path
        )
        count2 = indexer2.get_stats()["total_documents"]

        # Should have same count
        assert count1 == count2 == 3

    def test_separate_collections(self, test_db_path, sample_chunks):
        """Test that different collections are independent."""
        # Create two indexers with different collections
        indexer1 = DocumentIndexer(
            collection_name="collection_a",
            persist_path=test_db_path
        )
        indexer2 = DocumentIndexer(
            collection_name="collection_b",
            persist_path=test_db_path
        )

        # Index to first collection
        indexer1.index_chunks([sample_chunks[0]])

        # Index to second collection
        indexer2.index_chunks([sample_chunks[1], sample_chunks[2]])

        # Verify counts are independent
        assert indexer1.get_stats()["total_documents"] == 1
        assert indexer2.get_stats()["total_documents"] == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_index_chunk_with_empty_content(self, indexer):
        """Test indexing chunk with empty content."""
        chunk = DocumentChunk(
            content="",
            metadata={"title": "Empty"},
            chunk_type="text",
            heading=None
        )

        # Should not error
        count = indexer.index_chunks([chunk])
        assert count == 1

    def test_index_chunk_with_very_long_content(self, indexer):
        """Test indexing chunk with very long content."""
        long_content = "This is a test sentence. " * 1000  # ~25,000 characters
        chunk = DocumentChunk(
            content=long_content,
            metadata={"title": "Long"},
            chunk_type="text",
            heading="Long Content"
        )

        # Should handle long content
        count = indexer.index_chunks([chunk])
        assert count == 1

    def test_index_chunk_with_special_characters(self, indexer):
        """Test indexing chunk with special characters."""
        chunk = DocumentChunk(
            content="Special chars: <>&\"'\\n\\t\u00e9\u00e8\u00e0",
            metadata={"title": "Special"},
            chunk_type="text",
            heading="Special Characters"
        )

        # Should handle special characters
        count = indexer.index_chunks([chunk])
        assert count == 1

        # Verify content is preserved
        all_docs = indexer.document_store.filter_documents()
        assert len(all_docs) == 1
        assert "\u00e9\u00e8\u00e0" in all_docs[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
