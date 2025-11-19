"""
Base interface for embedding providers.

This module defines the abstract base class for embedding providers,
allowing the system to support multiple embedding models from different
providers (Sentence Transformers, OpenAI, Cohere, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding provider.

    Attributes:
        model_name: Name or identifier of the embedding model
        dimensions: Dimensionality of the embedding vectors
        max_tokens: Maximum number of tokens per text chunk
        batch_size: Batch size for processing multiple texts
        normalize: Whether to normalize embeddings to unit length
        api_key: Optional API key for cloud providers
        additional_params: Provider-specific parameters
    """
    model_name: str
    dimensions: int
    max_tokens: int = 512
    batch_size: int = 32
    normalize: bool = True
    api_key: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResult:
    """
    Result from embedding generation.

    Attributes:
        embeddings: List of embedding vectors
        model: Model name used
        dimensions: Vector dimensionality
        token_counts: Number of tokens in each text (if available)
    """
    embeddings: List[np.ndarray]
    model: str
    dimensions: int
    token_counts: Optional[List[int]] = None


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Each provider is responsible for:
    1. Loading/initializing the embedding model
    2. Generating embeddings for text
    3. Handling batching and rate limiting
    4. Normalizing embeddings if needed
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding provider.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._validate_config()
        self._model = None

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the embedding model.

        This might involve:
        - Loading model weights from disk
        - Connecting to an API
        - Initializing tokenizers

        Raises:
            ModelLoadError: If model loading fails
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of documents with batching.

        Args:
            documents: List of document texts
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingResult with all embeddings and metadata
        """
        if not self._model:
            self.load_model()

        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embeddings = self.embed_batch(batch)
            all_embeddings.extend(embeddings)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.config.model_name,
            dimensions=self.config.dimensions
        )

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings.

        Returns:
            Number of dimensions in embedding vectors
        """
        return self.config.dimensions

    def supports_batching(self) -> bool:
        """
        Check if provider supports batch processing.

        Returns:
            True if batching is supported
        """
        return True

    def get_max_batch_size(self) -> int:
        """
        Get maximum batch size supported.

        Returns:
            Maximum number of texts per batch
        """
        return self.config.batch_size

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.config.model_name,
            "dimensions": self.config.dimensions,
            "max_tokens": self.config.max_tokens,
            "normalize": self.config.normalize,
            "provider": self.__class__.__name__
        }

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources (unload model, close connections, etc.).
        """
        pass

    def __enter__(self):
        """Context manager entry: load model."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: cleanup resources."""
        self.cleanup()


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when model loading fails."""
    pass


class TokenLimitExceededError(EmbeddingError):
    """Raised when text exceeds token limit."""
    pass


class RateLimitError(EmbeddingError):
    """Raised when API rate limit is exceeded."""
    pass
