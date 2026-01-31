"""
Factory for creating embedding providers from configuration.

This module provides a factory pattern for instantiating embedding providers,
making it easy to switch between different providers via configuration.
"""

import logging
import os
from typing import Dict, Any, Optional, List
import numpy as np
import yaml

from .providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    SentenceTransformersProvider,
    OpenAIEmbeddingProvider,
    OPENAI_AVAILABLE
)

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers.

    Supports multiple embedding providers and configuration formats.
    """

    # Registry of available providers
    PROVIDERS = {
        "sentence-transformers": SentenceTransformersProvider,
        "openai": OpenAIEmbeddingProvider if OPENAI_AVAILABLE else None,
    }

    # Default configurations for each provider
    DEFAULTS = {
        "sentence-transformers": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "batch_size": 32,
            "normalize": True,
        },
        "openai": {
            "model_name": "text-embedding-3-small",
            "dimensions": 1536,
            "batch_size": 100,
            "normalize": True,
        },
    }

    @classmethod
    def create(
        cls,
        provider_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EmbeddingProvider:
        """
        Create an embedding provider from configuration.

        Args:
            provider_type: Type of provider ("sentence-transformers", "openai", etc.)
            config: Configuration dictionary (optional)
            **kwargs: Additional configuration as keyword arguments

        Returns:
            Initialized EmbeddingProvider instance

        Raises:
            ValueError: If provider type is unknown or not available

        Example:
            >>> provider = EmbeddingProviderFactory.create(
            ...     "openai",
            ...     config={"api_key": "sk-...", "model_name": "text-embedding-3-small"}
            ... )
            >>> provider.load_model()
        """
        # Normalize provider type
        provider_type = provider_type.lower().strip()

        # Check if provider is registered
        if provider_type not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available providers: {available}"
            )

        # Check if provider is available
        provider_class = cls.PROVIDERS[provider_type]
        if provider_class is None:
            raise ValueError(
                f"Provider '{provider_type}' is not available. "
                f"Install required dependencies (e.g., pip install openai)"
            )

        # Merge configuration sources (defaults < config dict < kwargs)
        final_config = cls._build_config(provider_type, config, kwargs)

        # Substitute environment variables
        final_config = cls._substitute_env_vars(final_config)

        # Create EmbeddingConfig object
        embedding_config = EmbeddingConfig(**final_config)

        # Instantiate provider
        logger.info(f"Creating {provider_type} embedding provider")
        provider = provider_class(embedding_config)

        return provider

    @classmethod
    def create_from_yaml(cls, yaml_path: str) -> EmbeddingProvider:
        """
        Create an embedding provider from YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Initialized EmbeddingProvider instance

        Example YAML:
            embedding:
              provider: openai
              model_name: text-embedding-3-small
              dimensions: 1536
              api_key: ${OPENAI_API_KEY}
              batch_size: 100
        """
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Extract embedding configuration
        embedding_config = config_data.get("embedding", {})

        if "provider" not in embedding_config:
            raise ValueError("Configuration must include 'provider' field")

        provider_type = embedding_config.pop("provider")

        return cls.create(provider_type, config=embedding_config)

    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> EmbeddingProvider:
        """
        Create an embedding provider from dictionary configuration.

        Args:
            config_dict: Dictionary with 'provider' key and provider config

        Returns:
            Initialized EmbeddingProvider instance

        Example:
            >>> config = {
            ...     "provider": "openai",
            ...     "model_name": "text-embedding-3-small",
            ...     "api_key": "sk-..."
            ... }
            >>> provider = EmbeddingProviderFactory.create_from_dict(config)
        """
        if "provider" not in config_dict:
            raise ValueError("Configuration must include 'provider' field")

        config_copy = config_dict.copy()
        provider_type = config_copy.pop("provider")

        return cls.create(provider_type, config=config_copy)

    @classmethod
    def _build_config(
        cls,
        provider_type: str,
        config: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build final configuration by merging defaults, config dict, and kwargs.

        Priority: kwargs > config > defaults
        """
        # Start with defaults
        final_config = cls.DEFAULTS.get(provider_type, {}).copy()

        # Merge config dict
        if config:
            final_config.update(config)

        # Merge kwargs (highest priority)
        final_config.update(kwargs)

        return final_config

    @classmethod
    def _substitute_env_vars(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variable placeholders in config values.

        Supports ${VAR_NAME} syntax.
        """
        result = {}

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable name
                var_name = value[2:-1]
                # Get from environment
                env_value = os.getenv(var_name)

                if env_value is None:
                    logger.warning(
                        f"Environment variable {var_name} not set for config key '{key}'"
                    )
                    result[key] = None
                else:
                    result[key] = env_value
            else:
                result[key] = value

        return result

    @classmethod
    def list_providers(cls) -> Dict[str, bool]:
        """
        List all registered providers and their availability.

        Returns:
            Dictionary mapping provider name to availability status
        """
        return {
            name: (provider_class is not None)
            for name, provider_class in cls.PROVIDERS.items()
        }

    @classmethod
    def get_provider_info(cls, provider_type: str) -> Dict[str, Any]:
        """
        Get information about a specific provider.

        Args:
            provider_type: Type of provider

        Returns:
            Dictionary with provider information
        """
        provider_type = provider_type.lower().strip()

        if provider_type not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider type: {provider_type}")

        provider_class = cls.PROVIDERS[provider_type]
        defaults = cls.DEFAULTS.get(provider_type, {})

        return {
            "name": provider_type,
            "available": provider_class is not None,
            "class": provider_class.__name__ if provider_class else None,
            "defaults": defaults,
        }


# Convenience function
def create_embedding_provider(
    provider_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Convenience function to create an embedding provider.

    Args:
        provider_type: Type of provider ("sentence-transformers", "openai", etc.)
        config: Configuration dictionary (optional)
        **kwargs: Additional configuration as keyword arguments

    Returns:
        Initialized EmbeddingProvider instance

    Example:
        >>> # Using sentence transformers (local, free)
        >>> provider = create_embedding_provider("sentence-transformers")
        >>> provider.load_model()
        >>>
        >>> # Using OpenAI (requires API key)
        >>> provider = create_embedding_provider(
        ...     "openai",
        ...     api_key="sk-...",
        ...     model_name="text-embedding-3-small"
        ... )
        >>> provider.load_model()
    """
    return EmbeddingProviderFactory.create(provider_type, config, **kwargs)


class EmbeddingProcessor:
    """
    High-level processor for generating embeddings from chunks.
    
    Combines factory functionality with batch processing and statistics.
    """
    
    def __init__(self, provider_type: str = "sentence-transformers", config: Optional[Dict[str, Any]] = None):
        """
        Initialize embedding processor.
        
        Args:
            provider_type: Type of embedding provider
            config: Configuration for the provider
        """
        self.provider = EmbeddingProviderFactory.create(provider_type, config)
        self.model_info = None
    
    def load_model(self):
        """Load the embedding model."""
        self.provider.load_model()
        self.model_info = self.provider.get_model_info()
        return self.model_info
    
    def embed_chunks(self, chunks: List[Any], sample_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk objects with content attribute
            sample_size: Number of chunks to process (None for all)
            
        Returns:
            List of embedding arrays
        """
        if sample_size is not None:
            chunks = chunks[:sample_size]
        
        embeddings = []
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            emb = self.provider.embed_text(chunk.content)
            embeddings.append(emb)
            logger.debug(f"Embedded chunk {i+1}/{len(chunks)}")
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def calculate_embedding_stats(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate statistics for generated embeddings.
        
        Args:
            embeddings: List of embedding arrays
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embeddings:
            return {}
        
        first_emb = embeddings[0]
        norms = [np.linalg.norm(emb) for emb in embeddings]
        
        return {
            "model_name": self.model_info["model_name"].split("/")[-1] if self.model_info else "unknown",
            "dimensions": self.model_info["dimensions"] if self.model_info else len(first_emb),
            "chunks_embedded": len(embeddings),
            "embedding_dtype": str(first_emb.dtype),
            "avg_embedding_norm": np.mean(norms),
            "embedding_norm_std": np.std(norms),
            "first_embedding_norm": norms[0]
        }
    
    def calculate_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate cosine similarity matrix between embeddings.
        
        Args:
            embeddings: List of embedding arrays
            
        Returns:
            2D numpy array with similarity scores
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def get_embedding_preview(self, embeddings: List[np.ndarray], num_dims: int = 10) -> Dict[str, Any]:
        """
        Get preview information for embeddings.
        
        Args:
            embeddings: List of embedding arrays
            num_dims: Number of dimensions to show in preview
            
        Returns:
            Dictionary with preview information
        """
        if not embeddings:
            return {}
        
        first_emb = embeddings[0]
        preview_values = first_emb[:num_dims]
        
        return {
            "preview_dimensions": preview_values.tolist(),
            "preview_string": ", ".join(f"{v:.4f}" for v in preview_values),
            "total_dimensions": len(first_emb),
            "first_norm": np.linalg.norm(first_emb)
        }
