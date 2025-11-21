"""
Unit tests for SentenceTransformersProvider.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.embedding.providers.base import (
    EmbeddingConfig,
    EmbeddingError,
    ModelLoadError
)
from src.embedding.providers.sentence_transformers import SentenceTransformersProvider


@pytest.fixture
def config():
    """Create a test configuration."""
    return EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        batch_size=32,
        normalize=True
    )


@pytest.fixture
def provider(config):
    """Create a SentenceTransformersProvider instance."""
    return SentenceTransformersProvider(config)


class TestInitialization:
    """Tests for provider initialization."""

    def test_initialization(self, config):
        """Test basic initialization."""
        provider = SentenceTransformersProvider(config)

        assert provider.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.config.dimensions == 384
        assert provider.config.batch_size == 32
        assert provider.config.normalize is True
        assert provider._model is None

    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        config = EmbeddingConfig(
            model_name="",
            dimensions=384
        )

        with pytest.raises(ValueError, match="model_name is required"):
            SentenceTransformersProvider(config)

    def test_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        config = EmbeddingConfig(
            model_name="test-model",
            dimensions=0
        )

        with pytest.raises(ValueError, match="dimensions must be positive"):
            SentenceTransformersProvider(config)

    def test_invalid_batch_size(self):
        """Test initialization with invalid batch size."""
        config = EmbeddingConfig(
            model_name="test-model",
            dimensions=384,
            batch_size=-1
        )

        with pytest.raises(ValueError, match="batch_size must be positive"):
            SentenceTransformersProvider(config)


class TestModelLoading:
    """Tests for model loading."""

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_load_model(self, mock_st, provider):
        """Test successful model loading."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()

        assert provider._model is not None
        mock_st.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_load_model_already_loaded(self, mock_st, provider):
        """Test loading model when already loaded."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()
        provider.load_model()  # Second call should not reload

        # Should only be called once
        assert mock_st.call_count == 1

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_load_model_dimension_mismatch(self, mock_st, provider):
        """Test model loading with dimension mismatch."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 512  # Different from config
        mock_st.return_value = mock_model

        provider.load_model()

        # Config should be updated
        assert provider.config.dimensions == 512

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_load_model_failure(self, mock_st, provider):
        """Test model loading failure."""
        mock_st.side_effect = Exception("Model not found")

        with pytest.raises(ModelLoadError, match="Failed to load model"):
            provider.load_model()


class TestEmbedText:
    """Tests for single text embedding."""

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_text(self, mock_st, provider):
        """Test embedding a single text."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_st.return_value = mock_model

        provider.load_model()
        embedding = provider.embed_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        mock_model.encode.assert_called_once_with(
            "Hello world",
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def test_embed_text_model_not_loaded(self, provider):
        """Test embedding without loading model."""
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            provider.embed_text("Hello world")

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_empty_text(self, mock_st, provider):
        """Test embedding empty text."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            provider.embed_text("")

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_whitespace_text(self, mock_st, provider):
        """Test embedding whitespace-only text."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            provider.embed_text("   ")

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_text_encoding_error(self, mock_st, provider):
        """Test embedding with encoding error."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_st.return_value = mock_model

        provider.load_model()

        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            provider.embed_text("Hello world")


class TestEmbedBatch:
    """Tests for batch text embedding."""

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_batch(self, mock_st, provider):
        """Test embedding multiple texts."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_st.return_value = mock_model

        provider.load_model()
        texts = ["Hello", "World", "Test"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        mock_model.encode.assert_called_once_with(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )

    def test_embed_batch_model_not_loaded(self, provider):
        """Test batch embedding without loading model."""
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            provider.embed_batch(["Hello", "World"])

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_empty_batch(self, mock_st, provider):
        """Test embedding empty list."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()
        embeddings = provider.embed_batch([])

        assert embeddings == []

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_batch_with_empty_texts(self, mock_st, provider):
        """Test embedding batch with some empty texts."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Only return embeddings for non-empty texts
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_st.return_value = mock_model

        provider.load_model()
        texts = ["Hello", "", "World"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        # Empty text should have zero vector
        assert np.allclose(embeddings[1], np.zeros(384))
        # Non-empty should have actual embeddings
        assert not np.allclose(embeddings[0], np.zeros(384))
        assert not np.allclose(embeddings[2], np.zeros(384))

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_batch_all_empty(self, mock_st, provider):
        """Test embedding batch with all empty texts."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()

        with pytest.raises(EmbeddingError, match="All input texts are empty"):
            provider.embed_batch(["", "  ", ""])

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_batch_encoding_error(self, mock_st, provider):
        """Test batch embedding with encoding error."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("Batch encoding failed")
        mock_st.return_value = mock_model

        provider.load_model()

        with pytest.raises(EmbeddingError, match="Failed to generate batch embeddings"):
            provider.embed_batch(["Hello", "World"])


class TestEmbedDocuments:
    """Tests for document embedding with batching."""

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_documents(self, mock_st, provider):
        """Test embedding documents with batching."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Return different batches
        mock_model.encode.side_effect = [
            np.random.rand(32, 384),  # First batch
            np.random.rand(18, 384),  # Second batch
        ]
        mock_st.return_value = mock_model

        provider.load_model()
        documents = [f"Document {i}" for i in range(50)]
        result = provider.embed_documents(documents)

        assert len(result.embeddings) == 50
        assert result.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert result.dimensions == 384
        # Should call encode twice (32 + 18 = 50)
        assert mock_model.encode.call_count == 2

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_embed_documents_auto_load(self, mock_st, provider):
        """Test that embed_documents auto-loads model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(5, 384)
        mock_st.return_value = mock_model

        # Don't load model explicitly
        documents = [f"Document {i}" for i in range(5)]
        result = provider.embed_documents(documents)

        assert len(result.embeddings) == 5
        assert provider._model is not None


class TestCleanup:
    """Tests for resource cleanup."""

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_cleanup(self, mock_st, provider):
        """Test cleanup unloads model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider.load_model()
        assert provider._model is not None

        provider.cleanup()
        assert provider._model is None

    def test_cleanup_without_loading(self, provider):
        """Test cleanup without loading model."""
        provider.cleanup()  # Should not raise error
        assert provider._model is None


class TestContextManager:
    """Tests for context manager interface."""

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_context_manager(self, mock_st, config):
        """Test using provider as context manager."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_st.return_value = mock_model

        with SentenceTransformersProvider(config) as provider:
            assert provider._model is not None
            embedding = provider.embed_text("Hello world")
            assert isinstance(embedding, np.ndarray)

        # Model should be cleaned up after context
        assert provider._model is None


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_embedding_dimension(self, provider):
        """Test getting embedding dimension."""
        assert provider.get_embedding_dimension() == 384

    def test_supports_batching(self, provider):
        """Test batching support."""
        assert provider.supports_batching() is True

    def test_get_max_batch_size(self, provider):
        """Test getting max batch size."""
        assert provider.get_max_batch_size() == 32

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_get_model_info(self, mock_st, provider):
        """Test getting model info."""
        info = provider.get_model_info()

        assert info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info["dimensions"] == 384
        assert info["provider"] == "SentenceTransformersProvider"

    @patch('src.embedding.providers.sentence_transformers.SentenceTransformer')
    def test_get_model_info_with_loaded_model(self, mock_st, provider):
        """Test getting model info when model is loaded."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        mock_st.return_value = mock_model

        provider.load_model()
        info = provider.get_model_info()

        assert info["actual_dimensions"] == 384
        assert info["max_seq_length"] == 512
