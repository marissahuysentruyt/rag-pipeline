#!/usr/bin/env python3
"""
Phase 5: Semantic Retrieval Demo

Demonstrates semantic search over indexed documents.
Shows how queries are embedded and matched to relevant chunks.

Uses: src.query.rag_pipeline.DesignSystemRAG (retrieval portion)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.table import Table

from demo.utils import DemoConsole
from demo.phase_4_index import DEMO_COLLECTION, DEMO_PERSIST_PATH

# Sample queries to demonstrate retrieval
SAMPLE_QUERIES = [
    "How do I create a primary button?",
    "What props does the Card component accept?",
    "Show me how to use Modal with focus management",
]


def run_demo(ensure_indexed: bool = True) -> dict:
    """
    Run the retrieval demo.

    Args:
        ensure_indexed: If True, run indexing first if needed

    Returns:
        Dict with retrieval results
    """
    console = DemoConsole()

    console.phase_header(
        phase_num=5,
        title="Semantic Retrieval",
        description=(
            "Search for relevant documentation using semantic similarity. "
            "Queries are embedded and matched against indexed chunks to find "
            "the most relevant content."
        )
    )

    # Check if we need to index first
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    console.info("Connecting to vector store...")
    document_store = ChromaDocumentStore(
        collection_name=DEMO_COLLECTION,
        persist_path=DEMO_PERSIST_PATH
    )

    doc_count = document_store.count_documents()
    console.console.print(f"  Found [green]{doc_count}[/green] indexed documents")
    console.console.print()

    if doc_count == 0 and ensure_indexed:
        console.warning("No documents indexed. Running Phase 4 first...")
        from demo.phase_4_index import run_demo as index_demo
        index_demo()
        console.transition("Continuing to retrieval")
        # Reconnect to get updated count
        document_store = ChromaDocumentStore(
            collection_name=DEMO_COLLECTION,
            persist_path=DEMO_PERSIST_PATH
        )
        doc_count = document_store.count_documents()

    # Initialize embedder for queries
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

    console.info("Initializing query embedder...")
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    with console.spinner("Loading embedder"):
        text_embedder.warm_up()

    # Initialize retriever
    retriever = ChromaEmbeddingRetriever(
        document_store=document_store,
        top_k=3
    )

    console.summary_table("Retrieval Configuration", {
        "Total indexed documents": doc_count,
        "Top K results": 3,
        "Embedding model": "all-MiniLM-L6-v2",
        "Collection": DEMO_COLLECTION,
    })

    # Run sample queries
    all_results = []

    for query in SAMPLE_QUERIES:
        console.console.print(f"\n[bold yellow]Query:[/bold yellow] {query}")

        with console.spinner("Embedding query and searching"):
            # Embed the query
            query_result = text_embedder.run(text=query)
            query_embedding = query_result["embedding"]

            # Retrieve documents
            retrieval_result = retriever.run(query_embedding=query_embedding)
            documents = retrieval_result["documents"]

        console.console.print(
            f"[green]Found {len(documents)} relevant chunks[/green]\n"
        )

        # Display results in a table
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Score", width=8)
        table.add_column("Component", width=15)
        table.add_column("Heading", width=20)
        table.add_column("Preview", width=40)

        for i, doc in enumerate(documents, 1):
            score = f"{doc.score:.3f}" if doc.score else "N/A"
            component = doc.meta.get("component", doc.meta.get("title", "Unknown"))
            heading = doc.meta.get("heading", "-")
            if heading and len(heading) > 18:
                heading = heading[:18] + "..."
            preview = doc.content[:80].replace("\n", " ")
            if len(preview) >= 80:
                preview = preview[:77] + "..."

            table.add_row(str(i), score, component, heading, preview)

        console.console.print(table)
        all_results.append({"query": query, "documents": documents})

    console.console.print()

    return {
        "results": all_results,
        "document_store": document_store,
        "retriever": retriever,
        "text_embedder": text_embedder,
    }


if __name__ == "__main__":
    run_demo()
