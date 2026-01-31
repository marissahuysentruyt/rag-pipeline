"""
Main entry point for embedding module.

Run with: python -m src.embedding
"""

import logging
from pathlib import Path
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.embedding import EmbeddingProcessor
from src.processing import MarkdownChunker, ChunkingConfig
from src.ingestion import DocumentProcessor


class EmbeddingConsole:
    """Simplified console output for embedding module."""
    
    def __init__(self):
        self.console = Console()
    
    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = "yellow"
        self.console.print()
        self.console.print(Panel(
            f"[bold {color}]Phase {phase_num}: {title}[/bold {color}]\n\n"
            f"[dim]{description}[/dim]",
            border_style=color,
            padding=(1, 2)
        ))
        self.console.print()
    
    def summary_table(self, title: str, data: Dict[str, Any]):
        """Display a summary statistics table."""
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, value in data.items():
            table.add_row(str(key), str(value))
        self.console.print(table)
        self.console.print()
    
    def info(self, message: str):
        """Print an info message."""
        self.console.print(f"[cyan]{message}[/cyan]")
    
    def success(self, message: str):
        """Print a success message."""
        self.console.print(f"[green]{message}[/green]")
    
    def warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"[yellow]{message}[/yellow]")
    
    @contextmanager
    def spinner(self, message: str):
        """Context manager for showing a spinner during operations."""
        with self.console.status(f"[bold green]{message}...", spinner="dots"):
            yield
    
    def display_embedding_preview(self, preview: Dict[str, Any]):
        """Display embedding preview information."""
        self.console.print("[bold]Sample Embedding (first 10 dimensions):[/bold]")
        self.console.print(f"  [{preview['preview_string']}, ...]")
        self.console.print()
    
    def display_similarity_matrix(self, embeddings: List, similarity_matrix: List[List[float]]):
        """Display semantic similarity matrix."""
        self.console.print("[bold]Semantic Similarity Matrix:[/bold]")
        self.console.print("[dim]Shows cosine similarity between chunk embeddings[/dim]\n")
        
        sim_table = Table(show_header=True, header_style="bold")
        sim_table.add_column("", style="cyan", width=8)
        for i in range(len(embeddings)):
            sim_table.add_column(f"C{i+1}", justify="right", width=7)
        
        for i, row in enumerate(similarity_matrix):
            table_row = [f"Chunk {i+1}"]
            for j, similarity in enumerate(row):
                # Color code: green for high similarity, yellow for medium
                if i == j:
                    table_row.append("[dim]1.000[/dim]")
                elif similarity > 0.7:
                    table_row.append(f"[green]{similarity:.3f}[/green]")
                elif similarity > 0.4:
                    table_row.append(f"[yellow]{similarity:.3f}[/yellow]")
                else:
                    table_row.append(f"{similarity:.3f}")
            sim_table.add_row(*table_row)
        
        self.console.print(sim_table)
        self.console.print()


def main():
    """Run embedding as a module with rich console output."""
    logging.basicConfig(level=logging.INFO)
    
    console = EmbeddingConsole()
    
    # Phase header
    console.phase_header(
        phase_num=3,
        title="Embedding Generation",
        description=(
            "Convert text chunks into dense vector representations. "
            "These embeddings capture semantic meaning, allowing similar "
            "concepts to be close together in vector space."
        )
    )
    
    # Load chunks from processing
    console.info("Loading chunks from processing...")
    
    # Load documents
    doc_processor = DocumentProcessor(min_chunk_size=200, max_chunk_size=1500)
    docs_path = Path("data/golden/docs")
    components_path = Path("data/golden/components")
    
    if not docs_path.exists():
        console.warning(f"Documentation path {docs_path} not found")
        console.info("Available methods:")
        console.info("  processor.embed_chunks(chunks)")
        console.info("  processor.calculate_embedding_stats(embeddings)")
        console.info("  processor.calculate_similarity_matrix(embeddings)")
        return
    
    # Load and process documents
    with console.spinner("Loading documents"):
        documents = doc_processor.process_batch(docs_path, components_path if components_path.exists() else None)
    
    # Chunk documents
    config = ChunkingConfig(
        min_chunk_size=200,
        max_chunk_size=1500,
        preserve_code_blocks=True
    )
    chunker = MarkdownChunker(config)
    
    with console.spinner("Chunking documents"):
        chunks = chunker.process_documents(documents)
    
    console.info(f"Loaded {len(chunks)} chunks for embedding")
    
    # Initialize embedding processor
    console.info("Initializing embedding processor...")
    processor = EmbeddingProcessor("sentence-transformers")
    
    with console.spinner("Loading model weights"):
        model_info = processor.load_model()
    
    console.console.print(f"  Model: [green]{model_info['model_name']}[/green]")
    console.console.print(f"  Dimensions: [green]{model_info['dimensions']}[/green]")
    console.console.print()
    
    # Generate embeddings for sample chunks
    sample_size = min(5, len(chunks))
    sample_chunks = chunks[:sample_size]
    
    console.console.print(f"[cyan]Generating embeddings for {sample_size} sample chunks...[/cyan]\n")
    
    embeddings = []
    for i, chunk in enumerate(sample_chunks, 1):
        preview = chunk.content[:50].replace("\n", " ")
        with console.spinner(f"Embedding chunk {i}: \"{preview}...\""):
            emb = processor.provider.embed_text(chunk.content)
            embeddings.append(emb)
    
    console.console.print()
    
    # Calculate and display statistics
    stats = processor.calculate_embedding_stats(embeddings)
    
    console.summary_table("Embedding Summary", {
        "Model": stats["model_name"],
        "Dimensions": stats["dimensions"],
        "Chunks embedded": stats["chunks_embedded"],
        "Embedding dtype": stats["embedding_dtype"],
        "Embedding norm": f"{stats['first_embedding_norm']:.4f}",
    })
    
    # Show sample embedding values
    preview = processor.get_embedding_preview(embeddings)
    console.display_embedding_preview(preview)
    
    # Show semantic similarity matrix
    similarity_matrix = processor.calculate_similarity_matrix(embeddings)
    console.display_similarity_matrix(embeddings, similarity_matrix.tolist())
    
    console.success("Embedding generation completed successfully!")


if __name__ == "__main__":
    main()
