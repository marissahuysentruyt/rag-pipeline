#!/usr/bin/env python3
"""
Phase 3: Embedding Generation Demo

Demonstrates how text chunks are converted into dense vector embeddings
using sentence transformers. Shows embedding dimensions and similarity.

Uses: src.embedding.factory.EmbeddingProviderFactory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.table import Table

from demo.utils import DemoConsole
from src.embedding.factory import EmbeddingProviderFactory


def run_demo(chunks: list = None) -> dict:
    """
    Run the embedding demo.

    Args:
        chunks: Pre-loaded chunks from phase 2 (optional)

    Returns:
        Dict with embeddings for next phase
    """
    console = DemoConsole()

    console.phase_header(
        phase_num=3,
        title="Embedding Generation",
        description=(
            "Convert text chunks into dense vector representations. "
            "These embeddings capture semantic meaning, allowing similar "
            "concepts to be close together in vector space."
        )
    )

    # Load chunks if not provided
    if chunks is None:
        console.info("Loading chunks from Phase 2...")
        from demo.phase_2_chunk import run_demo as chunk_demo
        result = chunk_demo()
        chunks = result["chunks"]
        console.transition("Continuing to embedding")

    # Create embedding provider using existing factory
    console.info("Initializing embedding provider...")
    provider = EmbeddingProviderFactory.create("sentence-transformers")

    with console.spinner("Loading model weights"):
        provider.load_model()

    model_info = provider.get_model_info()
    console.console.print(f"  Model: [green]{model_info['model_name']}[/green]")
    console.console.print(f"  Dimensions: [green]{model_info['dimensions']}[/green]")
    console.console.print()

    # Generate embeddings for sample chunks
    sample_size = min(5, len(chunks))
    sample_chunks = chunks[:sample_size]
    embeddings = []

    console.console.print(
        f"[cyan]Generating embeddings for {sample_size} sample chunks...[/cyan]\n"
    )

    for i, chunk in enumerate(sample_chunks, 1):
        preview = chunk.content[:50].replace("\n", " ")
        with console.spinner(f"Embedding chunk {i}: \"{preview}...\""):
            emb = provider.embed_text(chunk.content)
            embeddings.append(emb)

    console.console.print()

    # Summary
    console.summary_table("Embedding Summary", {
        "Model": model_info["model_name"].split("/")[-1],
        "Dimensions": model_info["dimensions"],
        "Chunks embedded": len(embeddings),
        "Embedding dtype": str(embeddings[0].dtype),
        "Embedding norm": f"{np.linalg.norm(embeddings[0]):.4f}",
    })

    # Show sample embedding values
    sample_emb = embeddings[0]
    console.console.print("[bold]Sample Embedding (first 10 dimensions):[/bold]")
    values = ", ".join(f"{v:.4f}" for v in sample_emb[:10])
    console.console.print(f"  [{values}, ...]")
    console.console.print()

    # Show semantic similarity matrix
    console.console.print("[bold]Semantic Similarity Matrix:[/bold]")
    console.console.print("[dim]Shows cosine similarity between chunk embeddings[/dim]\n")

    sim_table = Table(show_header=True, header_style="bold")
    sim_table.add_column("", style="cyan", width=8)
    for i in range(len(embeddings)):
        sim_table.add_column(f"C{i+1}", justify="right", width=7)

    for i, emb_i in enumerate(embeddings):
        row = [f"Chunk {i+1}"]
        for j, emb_j in enumerate(embeddings):
            # Cosine similarity
            similarity = np.dot(emb_i, emb_j) / (
                np.linalg.norm(emb_i) * np.linalg.norm(emb_j)
            )
            # Color code: green for high similarity, yellow for medium
            if i == j:
                row.append("[dim]1.000[/dim]")
            elif similarity > 0.7:
                row.append(f"[green]{similarity:.3f}[/green]")
            elif similarity > 0.4:
                row.append(f"[yellow]{similarity:.3f}[/yellow]")
            else:
                row.append(f"{similarity:.3f}")
        sim_table.add_row(*row)

    console.console.print(sim_table)
    console.console.print()

    return {
        "embeddings": embeddings,
        "chunks": chunks,
        "provider": provider,
        "model_info": model_info,
    }


if __name__ == "__main__":
    run_demo()
