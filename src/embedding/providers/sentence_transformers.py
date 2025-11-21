"""
Sentence Transformers embedding provider implementation.

This module provides a concrete implementation of the EmbeddingProvider
interface using the sentence-transformers library.
"""

import logging
from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from .base import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingError,
    ModelLoadError,
    TokenLimitExceededError
)

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """
    Embedding provider using Sentence Transformers models.

    Supports any model from the sentence-transformers library,
    including models from HuggingFace Hub.

    Example:
        >>> config = EmbeddingConfig(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     dimensions=384,
        ...     batch_size=32
        ... )
        >>> provider = SentenceTransformersProvider(config)
        >>> provider.load_model()
        >>> embedding = provider.embed_text("Hello world")
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the Sentence Transformers provider.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)
        self._model: Optional[SentenceTransformer] = None

    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.model_name:
            raise ValueError("model_name is required")

        if self.config.dimensions <= 0:
            raise ValueError("dimensions must be positive")

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def load_model(self) -> None:
        """
        Load the Sentence Transformers model.

        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model is not None:
            logger.debug(f"Model {self.config.model_name} already loaded")
            return

        try:
            logger.info(f"Loading Sentence Transformers model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)

            # Verify dimensions match
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.config.dimensions:
                logger.warning(
                    f"Model dimension mismatch: expected {self.config.dimensions}, "
                    f"got {actual_dim}. Updating config."
                )
                self.config.dimensions = actual_dim

            logger.info(
                f"Successfully loaded model {self.config.model_name} "
                f"(dimensions: {self.config.dimensions})"
            )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model {self.config.model_name}: {str(e)}"
            ) from e

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
            ModelLoadError: If model is not loaded
        """
        if self._model is None:
            raise ModelLoadError("Model not loaded. Call load_model() first.")

        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            # Generate embedding
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False
            )

            return embedding

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
            ModelLoadError: If model is not loaded
        """
        if self._model is None:
            raise ModelLoadError("Model not loaded. Call load_model() first.")

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
            # Generate embeddings for valid texts
            embeddings = self._model.encode(
                valid_texts,
                normalize_embeddings=self.config.normalize,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )

            # If we filtered out some texts, create a full result array
            if len(valid_texts) < len(texts):
                full_embeddings = []
                valid_idx = 0
                for i in range(len(texts)):
                    if i in valid_indices:
                        full_embeddings.append(embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        # Return zero vector for empty texts
                        full_embeddings.append(
                            np.zeros(self.config.dimensions, dtype=np.float32)
                        )
                return full_embeddings

            return [emb for emb in embeddings]

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {str(e)}"
            ) from e

    def cleanup(self) -> None:
        """
        Cleanup resources (unload model).
        """
        if self._model is not None:
            logger.info(f"Unloading model {self.config.model_name}")
            # Sentence transformers doesn't need explicit cleanup,
            # but we'll clear the reference
            self._model = None

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model metadata
        """
        info = super().get_model_info()

        if self._model is not None:
            info["actual_dimensions"] = self._model.get_sentence_embedding_dimension()
            info["max_seq_length"] = self._model.max_seq_length

        return info
