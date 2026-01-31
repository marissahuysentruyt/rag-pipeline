#!/usr/bin/env python3
"""
Phase 4: Vector Indexing Demo

Demonstrates how embedded chunks are stored in a vector database (Chroma)
for efficient similarity search.

Uses: src.ingestion.document_indexer.DocumentIndexer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.utils import DemoConsole
from src.ingestion.document_indexer import DocumentIndexer
from src.ingestion.document_processor import DocumentChunk
from src.processing.chunkers import ChunkType

# Demo-specific configuration
DEMO_COLLECTION = "golden_demo"
DEMO_PERSIST_PATH = "./data/demo_chroma_db"


def run_demo(chunks: list = None, documents: list = None) -> dict:
    """
    Run the indexing demo.

    Args:
        chunks: Pre-loaded chunks from phase 2 (optional)
        documents: Pre-loaded documents from phase 1 (optional, for metadata)

    Returns:
        Dict with indexer instance and stats
    """
    console = DemoConsole()

    console.phase_header(
        phase_num=4,
        title="Vector Indexing",
        description=(
            "Store embedded chunks in a vector database for efficient retrieval. "
            "Chroma provides persistent storage and fast similarity search."
        )
    )

    # Load chunks if needed
    if chunks is None:
        console.info("Loading chunks from Phase 2...")
        from demo.phase_2_chunk import run_demo as chunk_demo
        result = chunk_demo()
        chunks = result["chunks"]
        documents = result.get("documents")
        console.transition("Continuing to indexing")

    # Convert Chunk objects to DocumentChunk objects for the indexer
    console.info("Preparing chunks for indexing...")
    doc_chunks = []
    for chunk in chunks:
        # Map ChunkType to string
        chunk_type_str = "code" if chunk.chunk_type == ChunkType.CODE else "text"

        doc_chunk = DocumentChunk(
            content=chunk.content,
            metadata=chunk.metadata or {},
            chunk_type=chunk_type_str,
            heading=chunk.heading
        )
        doc_chunks.append(doc_chunk)

    console.console.print(f"  Prepared [green]{len(doc_chunks)}[/green] chunks")
    console.console.print()

    # Initialize the existing DocumentIndexer
    console.info(f"Initializing Chroma collection: [cyan]{DEMO_COLLECTION}[/cyan]")
    console.console.print(f"  Persist path: [dim]{DEMO_PERSIST_PATH}[/dim]")
    console.console.print()

    indexer = DocumentIndexer(
        collection_name=DEMO_COLLECTION,
        persist_path=DEMO_PERSIST_PATH
    )

    # Clear existing demo data for clean demo
    with console.spinner("Clearing existing demo data"):
        indexer.clear_index()

    # Index chunks
    with console.spinner(f"Indexing {len(doc_chunks)} chunks"):
        indexed_count = indexer.index_chunks(doc_chunks, batch_size=10)

    # Get stats
    stats = indexer.get_stats()

    console.summary_table("Indexing Summary", {
        "Chunks indexed": indexed_count,
        "Collection name": stats["collection_name"],
        "Embedding model": stats["embedding_model"].split("/")[-1],
        "Persist path": stats["persist_path"],
        "Total documents in store": stats["total_documents"],
    })

    # Show sample indexed document
    console.console.print("[bold]Sample Indexed Document:[/bold]")
    if doc_chunks:
        sample = doc_chunks[0]
        console.key_value("Content length", f"{len(sample.content)} chars")
        console.key_value("Chunk type", sample.chunk_type)
        console.key_value("Heading", sample.heading or "None")
        if sample.metadata:
            console.key_value("Component", sample.metadata.get("component", "N/A"))
            console.key_value("Category", sample.metadata.get("category", "N/A"))
    console.console.print()

    return {
        "indexer": indexer,
        "indexed_count": indexed_count,
        "stats": stats,
        "chunks": chunks,
    }


if __name__ == "__main__":
    run_demo()
