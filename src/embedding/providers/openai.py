"""
OpenAI embedding provider implementation.

This module provides a concrete implementation of the EmbeddingProvider
interface using OpenAI's embedding API.
"""

import logging
import time
from typing import List, Optional
import numpy as np

try:
    from openai import OpenAI
    from openai import RateLimitError as OpenAIRateLimitError
    from openai import APIError, AuthenticationError as OpenAIAuthError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingError,
    ModelLoadError,
    TokenLimitExceededError,
    RateLimitError
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI's embedding API.

    Supports OpenAI embedding models including:
    - text-embedding-3-small (1536 dimensions, faster, cheaper)
    - text-embedding-3-large (3072 dimensions, higher quality)
    - text-embedding-ada-002 (1536 dimensions, legacy)

    Example:
        >>> config = EmbeddingConfig(
        ...     model_name="text-embedding-3-small",
        ...     dimensions=1536,
        ...     api_key="sk-...",
        ...     batch_size=100
        ... )
        >>> provider = OpenAIEmbeddingProvider(config)
        >>> provider.load_model()
        >>> embedding = provider.embed_text("Hello world")
    """

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # API rate limits (requests per minute)
    RATE_LIMITS = {
        "text-embedding-3-small": 5000,
        "text-embedding-3-large": 5000,
        "text-embedding-ada-002": 3000,
    }

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the OpenAI embedding provider.

        Args:
            config: Embedding configuration with api_key required
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        super().__init__(config)
        self._client: Optional[OpenAI] = None
        self._request_count = 0
        self._last_request_time = time.time()

    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.model_name:
            raise ValueError("model_name is required")

        if not self.config.api_key:
            raise ValueError("api_key is required for OpenAI provider")

        if self.config.dimensions <= 0:
            raise ValueError("dimensions must be positive")

        # Validate model name
        if self.config.model_name not in self.MODEL_DIMENSIONS:
            logger.warning(
                f"Unknown model {self.config.model_name}. "
                f"Known models: {list(self.MODEL_DIMENSIONS.keys())}"
            )

        # Check dimension match for known models
        expected_dim = self.MODEL_DIMENSIONS.get(self.config.model_name)
        if expected_dim and expected_dim != self.config.dimensions:
            logger.warning(
                f"Dimension mismatch for {self.config.model_name}: "
                f"expected {expected_dim}, got {self.config.dimensions}. "
                f"Updating to {expected_dim}."
            )
            self.config.dimensions = expected_dim

    def load_model(self) -> None:
        """
        Initialize the OpenAI client.

        Raises:
            ModelLoadError: If client initialization fails
        """
        if self._client is not None:
            logger.debug("OpenAI client already initialized")
            return

        try:
            logger.info(f"Initializing OpenAI client for model: {self.config.model_name}")
            self._client = OpenAI(api_key=self.config.api_key)

            # Test the connection with a small request
            test_response = self._client.embeddings.create(
                model=self.config.model_name,
                input="test",
                encoding_format="float"
            )

            actual_dim = len(test_response.data[0].embedding)
            if actual_dim != self.config.dimensions:
                logger.warning(
                    f"Model dimension mismatch: expected {self.config.dimensions}, "
                    f"got {actual_dim}. Updating config."
                )
                self.config.dimensions = actual_dim

            logger.info(
                f"Successfully initialized OpenAI client "
                f"(model: {self.config.model_name}, dimensions: {self.config.dimensions})"
            )

        except OpenAIAuthError as e:
            raise ModelLoadError(
                f"Authentication failed. Check your API key: {str(e)}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize OpenAI client: {str(e)}"
            ) from e

    def _check_rate_limit(self) -> None:
        """
        Simple rate limiting to avoid API throttling.
        Resets counter every minute.
        """
        current_time = time.time()
        time_elapsed = current_time - self._last_request_time

        # Reset counter every 60 seconds
        if time_elapsed > 60:
            self._request_count = 0
            self._last_request_time = current_time

        # Get rate limit for this model
        rate_limit = self.RATE_LIMITS.get(self.config.model_name, 3000)

        # If we're approaching the limit, slow down
        if self._request_count >= rate_limit * 0.9:
            sleep_time = 60 - time_elapsed
            if sleep_time > 0:
                logger.warning(
                    f"Approaching rate limit ({self._request_count}/{rate_limit}). "
                    f"Sleeping for {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                self._request_count = 0
                self._last_request_time = time.time()

        self._request_count += 1

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
            ModelLoadError: If client is not initialized
        """
        if self._client is None:
            raise ModelLoadError("Client not initialized. Call load_model() first.")

        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            self._check_rate_limit()

            response = self._client.embeddings.create(
                model=self.config.model_name,
                input=text,
                encoding_format="float"
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Normalize if configured
            if self.config.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            return embedding

        except OpenAIRateLimitError as e:
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {str(e)}"
            ) from e
        except OpenAIAuthError as e:
            raise EmbeddingError(
                f"Authentication error: {str(e)}"
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding: {str(e)}"
            ) from e

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
            ModelLoadError: If client is not initialized
        """
        if self._client is None:
            raise ModelLoadError("Client not initialized. Call load_model() first.")

        if not texts:
            return []

        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            raise EmbeddingError("All input texts are empty")

        try:
            self._check_rate_limit()

            # OpenAI supports batching up to 2048 texts per request
            # But we'll use the configured batch_size for consistency
            all_embeddings = []

            for i in range(0, len(valid_texts), self.config.batch_size):
                batch = valid_texts[i:i+self.config.batch_size]

                response = self._client.embeddings.create(
                    model=self.config.model_name,
                    input=batch,
                    encoding_format="float"
                )

                # Extract embeddings in order
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in response.data
                ]

                # Normalize if configured
                if self.config.normalize:
                    batch_embeddings = [
                        emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
                        for emb in batch_embeddings
                    ]

                all_embeddings.extend(batch_embeddings)

            # If we filtered out some texts, create a full result array
            if len(valid_texts) < len(texts):
                full_embeddings = []
                valid_idx = 0
                for i in range(len(texts)):
                    if i in valid_indices:
                        full_embeddings.append(all_embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        # Return zero vector for empty texts
                        full_embeddings.append(
                            np.zeros(self.config.dimensions, dtype=np.float32)
                        )
                return full_embeddings

            return all_embeddings

        except OpenAIRateLimitError as e:
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {str(e)}"
            ) from e
        except OpenAIAuthError as e:
            raise EmbeddingError(
                f"Authentication error: {str(e)}"
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {str(e)}"
            ) from e

    def cleanup(self) -> None:
        """
        Cleanup resources (close client connection).
        """
        if self._client is not None:
            logger.info(f"Closing OpenAI client for {self.config.model_name}")
            # OpenAI client doesn't need explicit cleanup, but we'll clear the reference
            self._client = None
            self._request_count = 0

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model metadata
        """
        info = super().get_model_info()

        info["api_provider"] = "OpenAI"
        info["rate_limit"] = self.RATE_LIMITS.get(self.config.model_name, "unknown")
        info["supports_batching"] = True
        info["max_batch_size"] = 2048  # OpenAI's limit

        return info
