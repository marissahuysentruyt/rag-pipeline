"""
Simple RAG demo using sample design system documentation.
This demonstrates the end-to-end RAG pipeline flow.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample design system documentation
SAMPLE_DOCS = [
    {
        "content": """# Button Component

The Button component is used to trigger actions in the interface.

## Props
- `variant`: "primary" | "secondary" | "outline" - Button style variant
- `size`: "sm" | "md" | "lg" - Button size
- `disabled`: boolean - Whether the button is disabled
- `onClick`: function - Click event handler

## Usage
```jsx
<Button variant="primary" size="md" onClick={handleClick}>
  Click Me
</Button>
```

## Guidelines
- Use primary buttons for main actions
- Limit to one primary button per section
- Use secondary buttons for less important actions
""",
        "metadata": {
            "component": "Button",
            "category": "Actions",
            "framework": "React"
        }
    },
    {
        "content": """# Input Component

The Input component allows users to enter text data.

## Props
- `type`: "text" | "email" | "password" - Input type
- `placeholder`: string - Placeholder text
- `value`: string - Input value
- `onChange`: function - Change event handler
- `error`: boolean - Error state
- `helperText`: string - Helper or error text

## Usage
```jsx
<Input
  type="email"
  placeholder="Enter your email"
  value={email}
  onChange={(e) => setEmail(e.target.value)}
/>
```

## Guidelines
- Always include labels for accessibility
- Show error states clearly with helper text
- Use appropriate input types for better mobile keyboards
""",
        "metadata": {
            "component": "Input",
            "category": "Forms",
            "framework": "React"
        }
    },
    {
        "content": """# Card Component

The Card component is a container for grouping related content.

## Props
- `elevation`: 0 | 1 | 2 | 3 - Shadow depth
- `padding`: "none" | "sm" | "md" | "lg" - Internal padding
- `onClick`: function - Makes card clickable

## Usage
```jsx
<Card elevation={1} padding="md">
  <h3>Card Title</h3>
  <p>Card content goes here</p>
</Card>
```

## Guidelines
- Use cards to group related information
- Keep card content focused and scannable
- Use elevation to show hierarchy
""",
        "metadata": {
            "component": "Card",
            "category": "Layout",
            "framework": "React"
        }
    },
    {
        "content": """# Design Tokens - Spacing

Spacing tokens ensure consistent spacing throughout the design system.

## Spacing Scale
- `space-xs`: 4px - Minimal spacing
- `space-sm`: 8px - Small spacing
- `space-md`: 16px - Medium spacing (base)
- `space-lg`: 24px - Large spacing
- `space-xl`: 32px - Extra large spacing
- `space-2xl`: 48px - Section spacing

## Usage
Use these tokens instead of hard-coded pixel values to maintain consistency.

## Guidelines
- Use the spacing scale consistently across all components
- Prefer the base scale; create custom values only when necessary
- Use smaller spacing for related elements, larger for unrelated
""",
        "metadata": {
            "component": "Spacing",
            "category": "Design Tokens",
            "framework": "All"
        }
    }
]


def run_demo():
    """Run a simple RAG demo."""
    print("=" * 70)
    print("Simple RAG Pipeline Demo - Design System Documentation")
    print("=" * 70)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠ Error: ANTHROPIC_API_KEY not found")
        print("Please set your API key in .env file")
        return

    print("\n1. Setting up document store and embedder...")

    try:
        from haystack import Document
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
        from haystack.dataclasses import ChatMessage
        from haystack import Pipeline

        # Create document store
        document_store = InMemoryDocumentStore()

        # Create documents from sample data
        documents = [
            Document(content=doc["content"], meta=doc["metadata"])
            for doc in SAMPLE_DOCS
        ]

        print(f"   Created {len(documents)} sample documents")

        # Create embedder
        print("\n2. Generating embeddings...")
        print("   (This may take a moment on first run - downloading model)")

        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        doc_embedder.warm_up()

        # Embed documents
        embedded_docs = doc_embedder.run(documents)
        document_store.write_documents(embedded_docs["documents"])

        print(f"   ✓ Embedded and stored {len(documents)} documents")

        # Set up retrieval components
        print("\n3. Setting up retrieval pipeline...")

        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        text_embedder.warm_up()
        retriever = InMemoryEmbeddingRetriever(document_store=document_store)

        print("   ✓ Retrieval components ready")

        # Set up Claude generator
        print("\n4. Initializing Claude for response generation...")

        generator = AnthropicChatGenerator(
            model="claude-sonnet-4-5-20250929"
        )

        print("   ✓ Claude ready")

        # Demo queries
        queries = [
            "How do I use the Button component with different variants?",
            "What are the spacing tokens available?",
            "Show me how to create a form input with error handling"
        ]

        print("\n" + "=" * 70)
        print("Running Demo Queries")
        print("=" * 70)

        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 70}")
            print(f"Query {i}: {query}")
            print(f"{'─' * 70}")

            # Embed query
            query_embedding = text_embedder.run(query)

            # Retrieve relevant documents
            retrieved = retriever.run(
                query_embedding=query_embedding["embedding"],
                top_k=2
            )

            print(f"\nRetrieved {len(retrieved['documents'])} relevant documents:")
            for j, doc in enumerate(retrieved['documents'], 1):
                component = doc.meta.get('component', 'Unknown')
                category = doc.meta.get('category', 'Unknown')
                print(f"  {j}. {component} ({category})")

            # Build context from retrieved documents
            context = "\n\n".join([doc.content for doc in retrieved['documents']])

            # Generate response with Claude
            prompt = f"""You are a helpful assistant for a design system documentation.
Answer the user's question based on the following documentation context.
Be concise and include code examples if relevant.

Context:
{context}

Question: {query}

Answer:"""

            messages = [ChatMessage.from_user(prompt)]

            print("\nGenerating response with Claude...")
            response = generator.run(messages=messages)

            if response and 'replies' in response:
                answer = response['replies'][0].text
                print(f"\nResponse:\n{answer}")

        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        print("\nThis demo showed:")
        print("  ✓ Document embedding with sentence-transformers")
        print("  ✓ Semantic search and retrieval")
        print("  ✓ Response generation with Claude")
        print("  ✓ Context-aware answers from design system docs")

        print("\nNext steps:")
        print("  1. Add your own design system documentation")
        print("  2. Implement production vector store (Chroma)")
        print("  3. Build API or CLI interface")
        print("  4. Add metadata filtering by component/category")

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("  pip install sentence-transformers")
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
