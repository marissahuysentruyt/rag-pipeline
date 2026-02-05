"""
Main entry point for ingestion module.

Run with: python -m src.ingestion
"""

import logging
from pathlib import Path
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.ingestion.document_processor import DocumentProcessor


class IngestionConsole:
    """Simplified console output for ingestion module."""
    
    def __init__(self):
        self.console = Console()
    
    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = "cyan"
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
    
    def list_files(self, files: List[str], title: str = "Files"):
        """Display a list of files."""
        self.console.print(f"[bold]{title}:[/bold]")
        for f in files:
            self.console.print(f"  [dim]-[/dim] {f}")
        self.console.print()
    
    def key_value(self, key: str, value: Any, key_style: str = "cyan"):
        """Print a key-value pair."""
        self.console.print(f"  [{key_style}]{key}:[/{key_style}] {value}")


def main():
    """Run ingestion as a module with rich console output."""
    logging.basicConfig(level=logging.ERROR)
    
    console = IngestionConsole()
    
    # Phase header
    console.phase_header(
        phase_num=1,
        title="Document Ingestion",
        description=(
            "Load and parse markdown documentation files with YAML frontmatter. "
            "This phase extracts both content and structured metadata from each file."
        )
    )
    
    # Initialize processor
    processor = DocumentProcessor(min_chunk_size=200, max_chunk_size=1500)
    
    # Define paths
    docs_path = Path("data/golden/docs")
    components_path = Path("data/golden/components")
    
    if not docs_path.exists():
        console.warning(f"Documentation path {docs_path} not found")
        console.info("Available methods:")
        console.info("  processor.process_directory(path)")
        console.info("  processor.process_batch(docs_path, components_path)")
        console.info("  processor.calculate_stats(documents)")
        return
    
    # Find and list files
    md_files = list(docs_path.glob("*.md"))
    tsx_files = list(components_path.glob("*.tsx")) if components_path.exists() else []
    
    console.list_files([f.name for f in md_files], "Markdown Documentation Files")
    console.list_files([f.name for f in tsx_files], "TypeScript Component Files")
    
    # Process files with spinner
    with console.spinner("Processing documents"):
        documents = processor.process_batch(docs_path, components_path if components_path.exists() else None)
    
    # Show sample from first markdown document
    md_docs = [d for d in documents if d["path"].suffix == ".md"]
    if md_docs:
        sample = md_docs[0]
        console.console.print("[bold]Sample Document Metadata:[/bold]")
        for key, value in sample["doc"]["metadata"].items():
            console.key_value(key, value)
        console.console.print()
        
        # Content preview
        content_preview = sample["doc"]["content"][:400]
        console.sample_output(
            f"Content Preview ({sample['path'].name})",
            content_preview,
            syntax="markdown"
        )
    
    console.success("Ingestion completed successfully!")


if __name__ == "__main__":
    main()
