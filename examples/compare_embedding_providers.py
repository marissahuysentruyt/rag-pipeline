"""
Example: Compare different embedding providers.

This script demonstrates how to use the modular embedding provider architecture
to easily switch between different embedding models.

Usage:
    # Use local sentence transformers (free)
    python examples/compare_embedding_providers.py --provider sentence-transformers

    # Use OpenAI (requires API key)
    export OPENAI_API_KEY=sk-...
    python examples/compare_embedding_providers.py --provider openai

    # Compare both
    python examples/compare_embedding_providers.py --compare
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.factory import create_embedding_provider, EmbeddingProviderFactory
from src.embedding.providers import EmbeddingConfig


def benchmark_provider(provider_name: str, texts: list[str]) -> dict:
    """
    Benchmark an embedding provider.

    Args:
        provider_name: Name of the provider
        texts: List of texts to embed

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} Provider")
    print(f"{'='*60}")

    # Get provider info
    try:
        info = EmbeddingProviderFactory.get_provider_info(provider_name)
        if not info["available"]:
            print(f"❌ {provider_name} is not available. Install dependencies first.")
            return None
    except ValueError as e:
        print(f"❌ {str(e)}")
        return None

    # Create provider
    config = {}
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return None
        config["api_key"] = api_key

    try:
        print(f"\n1️⃣  Creating {provider_name} provider...")
        provider = create_embedding_provider(provider_name, config=config)
        print(f"   ✓ Provider created")
        print(f"   Model: {provider.config.model_name}")
        print(f"   Dimensions: {provider.config.dimensions}")

        print(f"\n2️⃣  Loading model...")
        start_time = time.time()
        provider.load_model()
        load_time = time.time() - start_time
        print(f"   ✓ Model loaded in {load_time:.2f}s")

        print(f"\n3️⃣  Embedding single text...")
        start_time = time.time()
        embedding = provider.embed_text(texts[0])
        single_time = time.time() - start_time
        print(f"   ✓ Generated embedding in {single_time:.3f}s")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.3f}")
        print(f"   Sample values: [{', '.join(f'{v:.3f}' for v in embedding[:5])}...]")

        print(f"\n4️⃣  Embedding batch of {len(texts)} texts...")
        start_time = time.time()
        embeddings = provider.embed_batch(texts)
        batch_time = time.time() - start_time
        print(f"   ✓ Generated {len(embeddings)} embeddings in {batch_time:.3f}s")
        print(f"   Avg time per text: {batch_time/len(texts):.3f}s")

        print(f"\n5️⃣  Computing similarity between first two texts...")
        emb1 = embeddings[0]
        emb2 = embeddings[1]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   Text 1: '{texts[0][:50]}...'")
        print(f"   Text 2: '{texts[1][:50]}...'")
        print(f"   Cosine similarity: {similarity:.3f}")

        provider.cleanup()

        return {
            "provider": provider_name,
            "model": provider.config.model_name,
            "dimensions": provider.config.dimensions,
            "load_time": load_time,
            "single_embedding_time": single_time,
            "batch_time": batch_time,
            "avg_time_per_text": batch_time / len(texts),
            "sample_embedding": embedding,
        }

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


def compare_providers(texts: list[str]):
    """Compare multiple embedding providers."""
    print("\n" + "="*60)
    print("COMPARING EMBEDDING PROVIDERS")
    print("="*60)

    results = {}

    # Test sentence transformers
    result = benchmark_provider("sentence-transformers", texts)
    if result:
        results["sentence-transformers"] = result

    # Test OpenAI
    result = benchmark_provider("openai", texts)
    if result:
        results["openai"] = result

    if len(results) < 2:
        print("\n⚠️  Need at least 2 providers to compare")
        return

    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print(f"\n{'Metric':<30} {'Sentence Trans.':<20} {'OpenAI':<20}")
    print("-" * 70)

    st_result = results.get("sentence-transformers")
    openai_result = results.get("openai")

    if st_result and openai_result:
        print(f"{'Model':<30} {st_result['model'][:18]:<20} {openai_result['model'][:18]:<20}")
        print(f"{'Dimensions':<30} {st_result['dimensions']:<20} {openai_result['dimensions']:<20}")
        print(f"{'Load time (s)':<30} {st_result['load_time']:<20.3f} {openai_result['load_time']:<20.3f}")
        print(f"{'Single embed time (s)':<30} {st_result['single_embedding_time']:<20.3f} {openai_result['single_embedding_time']:<20.3f}")
        print(f"{'Batch time (s)':<30} {st_result['batch_time']:<20.3f} {openai_result['batch_time']:<20.3f}")
        print(f"{'Avg time per text (s)':<30} {st_result['avg_time_per_text']:<20.3f} {openai_result['avg_time_per_text']:<20.3f}")

        print("\n💡 Key Takeaways:")
        print("   • Sentence Transformers: Free, local, no API limits")
        print("   • OpenAI: API-based, requires key, pay per use (~$0.00002/1K tokens)")


def main():
    parser = argparse.ArgumentParser(description="Compare embedding providers")
    parser.add_argument(
        "--provider",
        choices=["sentence-transformers", "openai"],
        help="Test a specific provider"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available providers"
    )
    args = parser.parse_args()

    # Sample texts to embed
    texts = [
        "The Button component allows users to trigger actions and make choices.",
        "Input components enable text entry from users in forms.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "The color palette includes primary, secondary, and accent colors.",
    ]

    print("Embedding Provider Comparison")
    print(f"Using {len(texts)} sample texts")

    if args.compare:
        compare_providers(texts)
    elif args.provider:
        benchmark_provider(args.provider, texts)
    else:
        # Show available providers
        print("\nAvailable providers:")
        for name, available in EmbeddingProviderFactory.list_providers().items():
            status = "✓" if available else "✗"
            print(f"  {status} {name}")

        print("\nUsage:")
        print("  python examples/compare_embedding_providers.py --provider sentence-transformers")
        print("  python examples/compare_embedding_providers.py --provider openai")
        print("  python examples/compare_embedding_providers.py --compare")


if __name__ == "__main__":
    main()
