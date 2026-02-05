"""
Main entry point for indexing module.

Run with: python -m src.ingestion.indexing
"""

import logging
from pathlib import Path
import sys
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


@contextmanager
def suppress_output():
    """Suppress stdout/stderr at the OS level for C libraries like MLX."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ingestion import DocumentIndexer
from src.processing import MarkdownChunker, ChunkingConfig
from src.ingestion import DocumentProcessor


class IndexingConsole:
    """Simplified console output for indexing module."""
    
    def __init__(self):
        self.console = Console()
    
    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = "magenta"
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
    
    def key_value(self, key: str, value: Any, key_style: str = "cyan"):
        """Print a key-value pair."""
        self.console.print(f"  [{key_style}]{key}:[/{key_style}] {value}")
    
    def display_sample_document(self, sample_info: Dict[str, Any]):
        """Display sample document information."""
        self.console.print("[bold]Sample Indexed Document:[/bold]")
        self.key_value("Content length", f"{sample_info['content_length']} chars")
        self.key_value("Chunk type", sample_info['chunk_type'])
        self.key_value("Heading", sample_info['heading'])
        if sample_info['component'] != "N/A":
            self.key_value("Component", sample_info['component'])
        if sample_info['category'] != "N/A":
            self.key_value("Category", sample_info['category'])
        if sample_info['title'] != "N/A":
            self.key_value("Title", sample_info['title'])
        self.console.print()


def main():
    """Run indexing as a module with rich console output."""
    logging.basicConfig(level=logging.ERROR)
    
    console = IndexingConsole()
    
    # Demo configuration
    DEMO_COLLECTION = "golden_demo"
    DEMO_PERSIST_PATH = "./data/demo_chroma_db"
    
    # Phase header
    console.phase_header(
        phase_num=4,
        title="Vector Indexing",
        description=(
            "Store embedded chunks in a vector database for efficient retrieval. "
            "Chroma provides persistent storage and fast similarity search."
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
        console.info("  indexer.index_chunks_from_pipeline(chunks)")
        console.info("  indexer.get_stats()")
        console.info("  indexer.get_sample_document_info(doc_chunks)")
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
    
    console.info(f"Loaded {len(chunks)} chunks for indexing")
    
    # Initialize the indexer
    with console.spinner(f"Initializing Chroma vector database: {DEMO_COLLECTION}"):
        with suppress_output():
            indexer = DocumentIndexer(
                collection_name=DEMO_COLLECTION,
                persist_path=DEMO_PERSIST_PATH
            )
    
    # Get stats
    stats = indexer.get_stats()
    
    console.info(f"Total documents in store: {stats['total_documents']}")
    
    console.success("Vector indexing completed successfully!")


if __name__ == "__main__":
    main()
