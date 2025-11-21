"""
Factory for creating embedding providers from configuration.

This module provides a factory pattern for instantiating embedding providers,
making it easy to switch between different providers via configuration.
"""

import logging
import os
from typing import Dict, Any, Optional
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
