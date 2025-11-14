"""
Basic setup verification script for RAG Pipeline.
Tests that all core components are installed and working.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import haystack
        print(f"✓ Haystack installed: v{haystack.__version__}")
    except ImportError as e:
        print(f"✗ Haystack import failed: {e}")
        return False

    try:
        import chromadb
        print(f"✓ ChromaDB installed: v{chromadb.__version__}")
    except ImportError as e:
        print(f"✗ ChromaDB import failed: {e}")
        return False

    try:
        import anthropic
        print(f"✓ Anthropic installed: v{anthropic.__version__}")
    except ImportError as e:
        print(f"✗ Anthropic import failed: {e}")
        return False

    try:
        from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
        print("✓ Anthropic-Haystack integration available")
    except ImportError as e:
        print(f"✗ Anthropic-Haystack integration failed: {e}")
        return False

    return True


def test_environment():
    """Test that required environment variables are set."""
    print("\nTesting environment configuration...")

    import os
    from dotenv import load_dotenv

    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv()
        print("✓ .env file found and loaded")
    else:
        print("⚠ .env file not found (using system environment variables)")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ ANTHROPIC_API_KEY found: {masked_key}")
        return True
    else:
        print("✗ ANTHROPIC_API_KEY not set")
        print("  Please copy .env.example to .env and add your API key")
        return False


def test_chroma_connection():
    """Test connection to ChromaDB."""
    print("\nTesting ChromaDB connection...")

    try:
        import chromadb
        from chromadb.config import Settings

        # Try to create an in-memory client first
        client = chromadb.Client()

        # Create a test collection
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"description": "Test collection for setup verification"}
        )

        # Add a test document
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test_1"]
        )

        # Query it back
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )

        if results['documents'][0][0] == "This is a test document":
            print("✓ ChromaDB in-memory client working")

            # Clean up
            client.delete_collection("test_collection")
            return True
        else:
            print("✗ ChromaDB query returned unexpected results")
            return False

    except Exception as e:
        print(f"✗ ChromaDB test failed: {e}")
        return False


def test_haystack_pipeline():
    """Test a basic Haystack pipeline."""
    print("\nTesting basic Haystack pipeline...")

    try:
        from haystack import Document, Pipeline
        from haystack.components.writers import DocumentWriter
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        # Create a simple document store
        doc_store = InMemoryDocumentStore()

        # Create some test documents
        docs = [
            Document(content="Button component allows users to trigger actions."),
            Document(content="Input component is used for text entry."),
            Document(content="Card component displays content in a container."),
        ]

        # Write documents to store
        doc_store.write_documents(docs)

        # Verify documents were stored
        stored_docs = doc_store.filter_documents()

        if len(stored_docs) == 3:
            print(f"✓ Haystack DocumentStore working ({len(stored_docs)} documents stored)")
            return True
        else:
            print(f"✗ Expected 3 documents, got {len(stored_docs)}")
            return False

    except Exception as e:
        print(f"✗ Haystack pipeline test failed: {e}")
        return False


def test_anthropic_integration():
    """Test Anthropic integration (requires API key)."""
    print("\nTesting Anthropic Claude integration...")

    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠ Skipping Anthropic test (no API key)")
        return True

    try:
        from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
        from haystack.dataclasses import ChatMessage

        # Create the generator (will use ANTHROPIC_API_KEY from environment)
        generator = AnthropicChatGenerator(
            model="claude-sonnet-4-5-20250929"  # Using Claude 4.5 Sonnet
        )

        # Test with a simple message
        messages = [
            ChatMessage.from_user("Reply with just the word 'success' and nothing else.")
        ]

        print("  Sending test message to Claude...")
        response = generator.run(messages=messages)

        if response and 'replies' in response:
            reply_text = response['replies'][0].text.strip().lower()
            if 'success' in reply_text:
                print("✓ Anthropic Claude responding correctly")
                return True
            else:
                print(f"✓ Anthropic Claude responding (reply: {reply_text[:50]}...)")
                return True
        else:
            print("✗ Unexpected response format from Claude")
            return False

    except Exception as e:
        print(f"✗ Anthropic integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Pipeline Setup Verification")
    print("=" * 60)

    results = {
        "imports": test_imports(),
        "environment": test_environment(),
        "chroma": test_chroma_connection(),
        "haystack": test_haystack_pipeline(),
        "anthropic": test_anthropic_integration(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! Your RAG pipeline setup is ready.")
        print("\nNext steps:")
        print("  1. Add design system documentation to data/raw/")
        print("  2. Run the ingestion pipeline (coming soon)")
        print("  3. Start querying your documentation")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
