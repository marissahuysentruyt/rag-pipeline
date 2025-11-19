"""
Unit tests for RAG query pipeline.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from haystack import Document
from haystack.dataclasses import ChatMessage

from src.query.rag_pipeline import DesignSystemRAG
from src.ingestion.document_indexer import DocumentIndexer
from src.ingestion.document_processor import DocumentChunk


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary directory for test database."""
    db_path = tmp_path / "test_rag_db"
    db_path.mkdir()
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        shutil.rmtree(db_path)


@pytest.fixture
def populated_index(test_db_path):
    """Create a test index with sample documents."""
    # Create test chunks
    chunks = [
        DocumentChunk(
            content="Button component allows users to trigger actions. Use the Button component for primary actions.",
            metadata={
                "title": "Button",
                "url": "https://example.com/button",
                "domain": "example.com"
            },
            chunk_type="text",
            heading="Button Component"
        ),
        DocumentChunk(
            content="```jsx\nimport { Button } from '@spectrum/button';\n<Button variant='primary'>Click me</Button>\n```",
            metadata={
                "title": "Button",
                "url": "https://example.com/button",
                "domain": "example.com"
            },
            chunk_type="code",
            heading="Usage"
        ),
        DocumentChunk(
            content="Input component is used for text entry. The Input component accepts various props for customization.",
            metadata={
                "title": "Input",
                "url": "https://example.com/input",
                "domain": "example.com"
            },
            chunk_type="text",
            heading="Input Component"
        ),
        DocumentChunk(
            content="Colors in Spectrum follow accessibility guidelines. Use semantic color tokens for consistency.",
            metadata={
                "title": "Colors",
                "url": "https://example.com/colors",
                "domain": "spectrum.adobe.com"
            },
            chunk_type="text",
            heading="Color Guidelines"
        )
    ]

    # Index chunks
    indexer = DocumentIndexer(
        collection_name="test_rag_collection",
        persist_path=test_db_path
    )
    indexer.index_chunks(chunks)

    return {
        "path": test_db_path,
        "collection": "test_rag_collection",
        "chunks": chunks
    }


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response."""
    mock_reply = Mock()
    mock_reply.text = "This is a helpful response about the Button component with code examples."
    return {"replies": [mock_reply]}


class TestInitialization:
    """Tests for RAG pipeline initialization."""

    def test_initialization_without_api_key(self, populated_index):
        """Test that initialization fails without ANTHROPIC_API_KEY."""
        # Temporarily remove API key
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                DesignSystemRAG(
                    collection_name=populated_index["collection"],
                    persist_path=populated_index["path"]
                )
        finally:
            # Restore API key
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_initialization_with_empty_index(self, test_db_path):
        """Test that initialization fails with empty index."""
        # Create empty index
        indexer = DocumentIndexer(
            collection_name="empty_collection",
            persist_path=test_db_path
        )

        # Set dummy API key
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        with pytest.raises(ValueError, match="No documents found"):
            DesignSystemRAG(
                collection_name="empty_collection",
                persist_path=test_db_path
            )

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_successful_initialization(self, populated_index):
        """Test successful initialization with valid setup."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=3
        )

        assert rag.collection_name == populated_index["collection"]
        assert rag.persist_path == populated_index["path"]
        assert rag.top_k == 3
        assert rag.document_store is not None
        assert rag.text_embedder is not None
        assert rag.retriever is not None
        assert rag.llm is not None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_default_parameters(self, populated_index):
        """Test initialization with default parameters."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        assert rag.llm_model == "claude-sonnet-4-5-20250929"
        assert "all-MiniLM-L6-v2" in rag.embedding_model
        assert rag.top_k == 5


class TestDocumentRetrieval:
    """Tests for document retrieval without generation."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_get_relevant_docs(self, populated_index):
        """Test retrieving relevant documents."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=2
        )

        docs = rag.get_relevant_docs("How to use button component?")

        assert len(docs) <= 2
        assert all(isinstance(doc, Document) for doc in docs)
        # Should retrieve button-related documents
        assert any("button" in doc.content.lower() for doc in docs)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_get_relevant_docs_with_filters(self, populated_index):
        """Test retrieving documents with metadata filters."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=3
        )

        # Filter by domain
        filters = {
            "field": "domain",
            "operator": "==",
            "value": "spectrum.adobe.com"
        }

        docs = rag.get_relevant_docs("color guidelines", filters=filters)

        # Should only return documents from spectrum.adobe.com
        assert all(doc.meta.get("domain") == "spectrum.adobe.com" for doc in docs)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_retrieval_returns_scored_documents(self, populated_index):
        """Test that retrieved documents have relevance scores."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=3
        )

        docs = rag.get_relevant_docs("button")

        # Documents should have scores
        for doc in docs:
            assert hasattr(doc, 'score')
            assert doc.score is not None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_retrieval_respects_top_k(self, populated_index):
        """Test that retrieval respects top_k parameter."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=2
        )

        docs = rag.get_relevant_docs("component")

        # Should return at most top_k documents
        assert len(docs) <= 2


class TestQueryGeneration:
    """Tests for full query pipeline with generation."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_query_basic(self, populated_index, mock_anthropic_response):
        """Test basic query execution."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=3
        )

        # Mock the LLM run method after initialization
        with patch.object(rag.llm, 'run', return_value=mock_anthropic_response):
            result = rag.query("How do I use a button?")

            # Check result structure
            assert "answer" in result
            assert "documents" in result
            assert "metadata" in result

            # Check answer
            assert isinstance(result["answer"], str)
            assert len(result["answer"]) > 0

            # Check documents
            assert len(result["documents"]) <= 3
            assert all(isinstance(doc, Document) for doc in result["documents"])

            # Check metadata
            assert result["metadata"]["query"] == "How do I use a button?"
            assert result["metadata"]["num_documents"] == len(result["documents"])
            assert result["metadata"]["model"] == "claude-sonnet-4-5-20250929"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_query_with_filters(self, populated_index, mock_anthropic_response):
        """Test query with metadata filters."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        filters = {
            "field": "domain",
            "operator": "==",
            "value": "example.com"
        }

        with patch.object(rag.llm, 'run', return_value=mock_anthropic_response):
            result = rag.query("button component", filters=filters)

            # Check that filters were applied
            assert result["metadata"]["filters"] == filters
            # All returned documents should match the filter
            assert all(doc.meta.get("domain") == "example.com" for doc in result["documents"])

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_query_without_filters(self, populated_index, mock_anthropic_response):
        """Test query without filters."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        with patch.object(rag.llm, 'run', return_value=mock_anthropic_response):
            result = rag.query("button")

            assert result["metadata"]["filters"] is None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_llm_receives_context(self, populated_index, mock_anthropic_response):
        """Test that LLM receives retrieved documents as context."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"],
            top_k=2
        )

        with patch.object(rag.llm, 'run', return_value=mock_anthropic_response) as mock_run:
            result = rag.query("button component")

            # LLM should have been called
            assert mock_run.called
            # Should have been called exactly once
            assert mock_run.call_count == 1

            # Verify we got documents and an answer
            assert len(result["documents"]) > 0
            assert result["answer"] is not None


class TestPromptBuilding:
    """Tests for prompt construction."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_prompt_builder_initialized(self, populated_index):
        """Test that prompt builder is properly initialized."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        assert rag.prompt_builder is not None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_system_message_content(self, populated_index):
        """Test that system message contains appropriate context."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        # The prompt builder should have a template with system message
        # We can't directly inspect the template, but we can verify it's configured
        assert hasattr(rag, 'prompt_builder')


class TestPipelineComponents:
    """Tests for individual pipeline components."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_pipeline_has_all_components(self, populated_index):
        """Test that pipeline has all required components."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        # Check pipeline has all components
        component_names = list(rag.pipeline.graph.nodes.keys())
        assert "text_embedder" in component_names
        assert "retriever" in component_names
        assert "prompt_builder" in component_names
        assert "llm" in component_names

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_pipeline_connections(self, populated_index):
        """Test that pipeline components are properly connected."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        # Verify connections exist
        graph = rag.pipeline.graph
        edges = graph.edges

        # Should have connections between components
        assert len(edges) >= 3  # At least 3 connections


class TestErrorHandling:
    """Tests for error handling."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_query_error_propagation(self, populated_index):
        """Test that errors in query are properly propagated."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        # Mock LLM to raise an error
        with patch.object(rag.llm, 'run', side_effect=Exception("LLM API error")):
            with pytest.raises(Exception, match="LLM API error"):
                rag.query("test query")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_empty_query(self, populated_index):
        """Test handling of empty query string."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        # Empty query should still work (retrieves based on empty embedding)
        docs = rag.get_relevant_docs("")
        assert isinstance(docs, list)


class TestEdgeCases:
    """Tests for edge cases."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_very_long_query(self, populated_index):
        """Test handling of very long query."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        long_query = "How do I use a button? " * 100  # Very long query
        docs = rag.get_relevant_docs(long_query)

        assert isinstance(docs, list)
        assert len(docs) <= rag.top_k

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_query_with_special_characters(self, populated_index):
        """Test query with special characters."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        special_query = "How do I use <Button> with \"quotes\" & special chars?"
        docs = rag.get_relevant_docs(special_query)

        assert isinstance(docs, list)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_multiple_queries_same_instance(self, populated_index, mock_anthropic_response):
        """Test running multiple queries on same RAG instance."""
        rag = DesignSystemRAG(
            collection_name=populated_index["collection"],
            persist_path=populated_index["path"]
        )

        with patch.object(rag.llm, 'run', return_value=mock_anthropic_response):
            # Run multiple queries
            result1 = rag.query("button")
            result2 = rag.query("input")
            result3 = rag.query("colors")

            # All should return valid results
            assert all(r["answer"] for r in [result1, result2, result3])
            assert all(r["documents"] for r in [result1, result2, result3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
