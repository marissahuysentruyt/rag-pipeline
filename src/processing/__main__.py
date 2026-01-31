"""
Main entry point for processing module.

Run with: python -m src.processing
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
from rich.syntax import Syntax
from rich.table import Table

from src.processing import MarkdownChunker, ChunkingConfig
from src.ingestion import DocumentProcessor


class ProcessingConsole:
    """Simplified console output for processing module."""
    
    def __init__(self):
        self.console = Console()
    
    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = "green"
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
    
    def sample_output(self, title: str, content: str, syntax: Optional[str] = None):
        """Display sample output in a panel."""
        if syntax:
            renderable = Syntax(content, syntax, theme="monokai", line_numbers=False)
        else:
            renderable = content
        self.console.print(Panel(
            renderable,
            title=f"[bold]Sample: {title}[/bold]",
            border_style="dim"
        ))
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
    
    def display_chunks(self, chunks: List[Dict]):
        """Display sample chunks."""
        self.console.print("[bold]Sample Chunks:[/bold]\n")
        for chunk in chunks:
            self.console.print(
                f"[cyan]Chunk {chunk['index']}[/cyan] "
                f"([yellow]{chunk['type']}[/yellow], {chunk['size']} chars)"
            )
            self.console.print(f"  Heading: [green]{chunk['heading']}[/green]")
            self.console.print(f"  Preview: {chunk['preview']}")
            self.console.print()


def main():
    """Run processing as a module with rich console output."""
    logging.basicConfig(level=logging.INFO)
    
    console = ProcessingConsole()
    
    # Phase header
    console.phase_header(
        phase_num=2,
        title="Document Chunking",
        description=(
            "Split documents into semantic chunks that preserve context. "
            "The chunker respects markdown structure, keeps code blocks intact, "
            "and maintains section headings for better retrieval."
        )
    )
    
    # Load documents from ingestion
    console.info("Loading documents from ingestion...")
    doc_processor = DocumentProcessor(min_chunk_size=200, max_chunk_size=1500)
    
    docs_path = Path("data/golden/docs")
    components_path = Path("data/golden/components")
    
    if not docs_path.exists():
        console.warning(f"Documentation path {docs_path} not found")
        console.info("Available methods:")
        console.info("  chunker.process_documents(documents)")
        console.info("  chunker.calculate_chunk_stats(chunks)")
        console.info("  chunker.get_sample_chunks(chunks)")
        return
    
    # Load documents
    with console.spinner("Loading documents"):
        documents = doc_processor.process_batch(docs_path, components_path if components_path.exists() else None)
    
    console.info(f"Loaded {len(documents)} documents for chunking")
    
    # Initialize chunker
    console.info("Initializing MarkdownChunker...")
    config = ChunkingConfig(
        min_chunk_size=200,
        max_chunk_size=1500,
        preserve_code_blocks=True
    )
    chunker = MarkdownChunker(config)
    
    # Process documents
    with console.spinner("Chunking documents"):
        chunks = chunker.process_documents(documents)
    
    # Calculate and display statistics
    stats = chunker.calculate_chunk_stats(chunks)
    
    console.summary_table("Chunking Summary", {
        "Total chunks created": stats['total_chunks'],
        "Text chunks": stats['text_chunks'],
        "Code-containing chunks": stats['code_chunks'],
        "Avg chunk size": f"{stats['avg_chunk_size']:,} chars",
        "Min chunk size": f"{stats['min_chunk_size']:,} chars",
        "Max chunk size": f"{stats['max_chunk_size']:,} chars",
    })
    
    # Show sample chunks
    samples = chunker.get_sample_chunks(chunks)
    console.display_chunks(samples)
    
    console.success("Chunking completed successfully!")


if __name__ == "__main__":
    main()
