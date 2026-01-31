#!/usr/bin/env python3
"""
Phase 2: Document Chunking Demo

Demonstrates how documents are split into semantic chunks while
preserving structure like code blocks, headings, and sections.

Uses: src.processing.chunkers.MarkdownChunker
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.utils import DemoConsole
from src.processing.chunkers import MarkdownChunker, ChunkingConfig, ChunkType


def run_demo(documents: list = None, data_path: Path = None) -> dict:
    """
    Run the chunking demo.

    Args:
        documents: Pre-loaded documents from phase 1 (optional)
        data_path: Path to load documents if not provided

    Returns:
        Dict with chunks for next phase
    """
    console = DemoConsole()

    console.phase_header(
        phase_num=2,
        title="Document Chunking",
        description=(
            "Split documents into semantic chunks that preserve context. "
            "The chunker respects markdown structure, keeps code blocks intact, "
            "and maintains section headings for better retrieval."
        )
    )

    # If no documents provided, load them from phase 1
    if documents is None:
        console.info("Loading documents from Phase 1...")
        from demo.phase_1_ingest import run_demo as ingest_demo
        result = ingest_demo(data_path)
        documents = result["documents"]
        console.transition("Continuing to chunking")

    # Initialize the existing MarkdownChunker
    console.info("Initializing MarkdownChunker...")
    config = ChunkingConfig(
        min_chunk_size=200,
        max_chunk_size=1500,
        preserve_code_blocks=True
    )
    chunker = MarkdownChunker(config)

    # Process documents
    all_chunks = []
    chunk_stats = {"text": 0, "code": 0}

    for doc_info in documents:
        doc = doc_info["doc"]
        filename = doc_info["path"].name

        with console.spinner(f"Chunking {filename}"):
            chunks = chunker.chunk_text(doc["content"], doc["metadata"])
            all_chunks.extend(chunks)

            for chunk in chunks:
                if chunk.chunk_type == ChunkType.CODE:
                    chunk_stats["code"] += 1
                else:
                    chunk_stats["text"] += 1

    # Summary statistics
    chunk_sizes = [len(c.content) for c in all_chunks]
    console.summary_table("Chunking Summary", {
        "Total chunks created": len(all_chunks),
        "Text chunks": chunk_stats["text"],
        "Code-containing chunks": chunk_stats["code"],
        "Avg chunk size": f"{sum(chunk_sizes) // len(all_chunks):,} chars" if all_chunks else "0",
        "Min chunk size": f"{min(chunk_sizes):,} chars" if chunk_sizes else "0",
        "Max chunk size": f"{max(chunk_sizes):,} chars" if chunk_sizes else "0",
    })

    # Show sample chunks
    console.console.print("[bold]Sample Chunks:[/bold]\n")
    for i, chunk in enumerate(all_chunks[:3], 1):
        chunk_type_label = "CODE" if chunk.chunk_type == ChunkType.CODE else "TEXT"
        console.console.print(
            f"[cyan]Chunk {i}[/cyan] "
            f"([yellow]{chunk_type_label}[/yellow], {len(chunk.content)} chars)"
        )
        console.console.print(f"  Heading: [green]{chunk.heading or 'None'}[/green]")

        # Show preview
        preview = chunk.content[:150].replace("\n", " ")
        console.console.print(f"  Preview: {preview}...")
        console.console.print()

    return {"chunks": all_chunks, "config": config, "documents": documents}


if __name__ == "__main__":
    run_demo()
