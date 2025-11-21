"""
Unit tests for OpenAIEmbeddingProvider.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.embedding.providers.base import (
    EmbeddingConfig,
    EmbeddingError,
    ModelLoadError,
    RateLimitError
)

# Test if OpenAI is available
try:
    from src.embedding.providers.openai import OpenAIEmbeddingProvider, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIEmbeddingProvider = None


@pytest.fixture
def config():
    """Create a test configuration."""
    return EmbeddingConfig(
        model_name="text-embedding-3-small",
        dimensions=1536,
        batch_size=100,
        normalize=True,
        api_key="sk-test-key-123"
    )


@pytest.fixture
def provider(config):
    """Create an OpenAIEmbeddingProvider instance."""
    if not OPENAI_AVAILABLE:
        pytest.skip("OpenAI library not installed")
    return OpenAIEmbeddingProvider(config)


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestInitialization:
    """Tests for provider initialization."""

    def test_initialization(self, config):
        """Test basic initialization."""
        provider = OpenAIEmbeddingProvider(config)

        assert provider.config.model_name == "text-embedding-3-small"
        assert provider.config.dimensions == 1536
        assert provider.config.batch_size == 100
        assert provider.config.normalize is True
        assert provider.config.api_key == "sk-test-key-123"
        assert provider._client is None

    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        config = EmbeddingConfig(
            model_name="",
            dimensions=1536,
            api_key="sk-test"
        )

        with pytest.raises(ValueError, match="model_name is required"):
            OpenAIEmbeddingProvider(config)

    def test_missing_api_key(self):
        """Test initialization without API key."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            dimensions=1536,
            api_key=None
        )

        with pytest.raises(ValueError, match="api_key is required"):
            OpenAIEmbeddingProvider(config)

    def test_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            dimensions=0,
            api_key="sk-test"
        )

        with pytest.raises(ValueError, match="dimensions must be positive"):
            OpenAIEmbeddingProvider(config)

    def test_dimension_correction(self):
        """Test automatic dimension correction for known models."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            dimensions=512,  # Wrong dimension
            api_key="sk-test"
        )

        provider = OpenAIEmbeddingProvider(config)
        # Should be corrected to 1536
        assert provider.config.dimensions == 1536

    def test_large_model_dimensions(self):
        """Test correct dimensions for large model."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-large",
            dimensions=3072,
            api_key="sk-test"
        )

        provider = OpenAIEmbeddingProvider(config)
        assert provider.config.dimensions == 3072


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestModelLoading:
    """Tests for model loading."""

    @patch('src.embedding.providers.openai.OpenAI')
    def test_load_model(self, mock_openai_class, provider):
        """Test successful model loading."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock the test embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()

        assert provider._client is not None
        mock_openai_class.assert_called_once_with(api_key="sk-test-key-123")
        mock_client.embeddings.create.assert_called_once()

    @patch('src.embedding.providers.openai.OpenAI')
    def test_load_model_already_loaded(self, mock_openai_class, provider):
        """Test loading model when already loaded."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()
        provider.load_model()  # Second call

        # Should only initialize once
        assert mock_openai_class.call_count == 1

    @patch('src.embedding.providers.openai.OpenAI')
    def test_load_model_auth_error(self, mock_openai_class, provider):
        """Test model loading with authentication error."""
        from openai import AuthenticationError

        mock_openai_class.side_effect = AuthenticationError(
            message="Invalid API key",
            response=Mock(status_code=401),
            body=None
        )

        with pytest.raises(ModelLoadError, match="Authentication failed"):
            provider.load_model()


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestEmbedText:
    """Tests for single text embedding."""

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_text(self, mock_openai_class, provider):
        """Test embedding a single text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock responses
        test_embedding = np.random.rand(1536).tolist()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=test_embedding)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()
        embedding = provider.embed_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536
        # Check normalization was applied
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=0.01)

    def test_embed_text_model_not_loaded(self, provider):
        """Test embedding without loading model."""
        with pytest.raises(ModelLoadError, match="Client not initialized"):
            provider.embed_text("Hello world")

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_empty_text(self, mock_openai_class, provider):
        """Test embedding empty text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            provider.embed_text("")

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_text_rate_limit_error(self, mock_openai_class, provider):
        """Test embedding with rate limit error."""
        from openai import RateLimitError as OpenAIRateLimitError

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call succeeds (load_model test)
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]

        # Second call fails with rate limit
        mock_client.embeddings.create.side_effect = [
            mock_response,
            OpenAIRateLimitError(
                message="Rate limit exceeded",
                response=Mock(status_code=429),
                body=None
            )
        ]

        provider.load_model()

        with pytest.raises(RateLimitError, match="rate limit exceeded"):
            provider.embed_text("Hello")


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestEmbedBatch:
    """Tests for batch text embedding."""

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_batch(self, mock_openai_class, provider):
        """Test embedding multiple texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock responses
        test_embeddings = [np.random.rand(1536).tolist() for _ in range(3)]
        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in test_embeddings]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()
        texts = ["Hello", "World", "Test"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(len(emb) == 1536 for emb in embeddings)

    def test_embed_batch_model_not_loaded(self, provider):
        """Test batch embedding without loading model."""
        with pytest.raises(ModelLoadError, match="Client not initialized"):
            provider.embed_batch(["Hello", "World"])

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_empty_batch(self, mock_openai_class, provider):
        """Test embedding empty list."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()
        embeddings = provider.embed_batch([])

        assert embeddings == []

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_batch_with_empty_texts(self, mock_openai_class, provider):
        """Test embedding batch with some empty texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock responses - only for non-empty texts
        test_embeddings = [np.random.rand(1536).tolist() for _ in range(2)]
        mock_response_load = Mock()
        mock_response_load.data = [Mock(embedding=[0.1] * 1536)]
        mock_response_batch = Mock()
        mock_response_batch.data = [Mock(embedding=emb) for emb in test_embeddings]

        mock_client.embeddings.create.side_effect = [
            mock_response_load,
            mock_response_batch
        ]

        provider.load_model()
        texts = ["Hello", "", "World"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        # Empty text should have zero vector
        assert np.allclose(embeddings[1], np.zeros(1536))
        # Non-empty should have actual embeddings
        assert not np.allclose(embeddings[0], np.zeros(1536))
        assert not np.allclose(embeddings[2], np.zeros(1536))

    @patch('src.embedding.providers.openai.OpenAI')
    def test_embed_batch_large(self, mock_openai_class, provider):
        """Test embedding large batch (tests batching logic)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create responses for multiple batches
        def create_batch_response(batch_size):
            return Mock(
                data=[Mock(embedding=np.random.rand(1536).tolist()) for _ in range(batch_size)]
            )

        # First call for load_model
        mock_client.embeddings.create.side_effect = [
            create_batch_response(1),  # load_model test
            create_batch_response(100),  # first batch
            create_batch_response(50),   # second batch
        ]

        provider.load_model()

        # Create 150 texts to test batching
        texts = [f"Text {i}" for i in range(150)]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 150
        # Should have made 3 calls total (1 for load, 2 for batches)
        assert mock_client.embeddings.create.call_count == 3


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestCleanup:
    """Tests for resource cleanup."""

    @patch('src.embedding.providers.openai.OpenAI')
    def test_cleanup(self, mock_openai_class, provider):
        """Test cleanup clears client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()
        assert provider._client is not None

        provider.cleanup()
        assert provider._client is None

    def test_cleanup_without_loading(self, provider):
        """Test cleanup without loading model."""
        provider.cleanup()  # Should not raise error
        assert provider._client is None


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestContextManager:
    """Tests for context manager interface."""

    @patch('src.embedding.providers.openai.OpenAI')
    def test_context_manager(self, mock_openai_class, config):
        """Test using provider as context manager."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        test_embedding = np.random.rand(1536).tolist()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=test_embedding)]
        mock_client.embeddings.create.return_value = mock_response

        with OpenAIEmbeddingProvider(config) as provider:
            assert provider._client is not None
            embedding = provider.embed_text("Hello world")
            assert isinstance(embedding, np.ndarray)

        # Client should be cleaned up after context
        assert provider._client is None


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_embedding_dimension(self, provider):
        """Test getting embedding dimension."""
        assert provider.get_embedding_dimension() == 1536

    def test_supports_batching(self, provider):
        """Test batching support."""
        assert provider.supports_batching() is True

    def test_get_max_batch_size(self, provider):
        """Test getting max batch size."""
        assert provider.get_max_batch_size() == 100

    @patch('src.embedding.providers.openai.OpenAI')
    def test_get_model_info(self, mock_openai_class, provider):
        """Test getting model info."""
        info = provider.get_model_info()

        assert info["model_name"] == "text-embedding-3-small"
        assert info["dimensions"] == 1536
        assert info["provider"] == "OpenAIEmbeddingProvider"
        assert info["api_provider"] == "OpenAI"
        assert info["supports_batching"] is True
        assert info["max_batch_size"] == 2048
        assert info["rate_limit"] == 5000


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestRateLimiting:
    """Tests for rate limiting logic."""

    @patch('src.embedding.providers.openai.OpenAI')
    @patch('time.sleep')
    def test_rate_limit_checking(self, mock_sleep, mock_openai_class, provider):
        """Test that rate limiting triggers when threshold reached."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        test_embedding = np.random.rand(1536).tolist()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=test_embedding)]
        mock_client.embeddings.create.return_value = mock_response

        provider.load_model()

        # Simulate many requests to trigger rate limit
        provider._request_count = 4500  # 90% of 5000
        provider._last_request_time = time.time()

        provider.embed_text("Test")

        # Should have triggered sleep
        mock_sleep.assert_called_once()


# Integration test marker
@pytest.mark.integration
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not installed")
class TestOpenAIIntegration:
    """
    Integration tests with real OpenAI API.

    These tests are skipped by default. Run with:
    pytest tests/test_openai_provider.py -m integration

    Requires OPENAI_API_KEY environment variable.
    """

    def test_real_api_small_model(self):
        """Test with real OpenAI API (small model)."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            dimensions=1536,
            api_key=api_key,
            normalize=True
        )

        with OpenAIEmbeddingProvider(config) as provider:
            embedding = provider.embed_text("Hello world")

            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 1536
            assert np.isclose(np.linalg.norm(embedding), 1.0, atol=0.01)
