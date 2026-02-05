"""
Main entry point for generation module.

Run with: python -m src.generation
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
from rich.markdown import Markdown
from rich.table import Table

from src.generation import RAGGenerator


class GenerationConsole:
    """Simplified console output for generation module."""
    
    def __init__(self):
        self.console = Console()
    
    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = "red"
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
    
    def error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]{message}[/red]")
    
    @contextmanager
    def spinner(self, message: str):
        """Context manager for showing a spinner during operations."""
        with self.console.status(f"[bold green]{message}...", spinner="dots"):
            yield
    
    def display_context_preview(self, context: str):
        """Display context preview."""
        self.console.print("[bold]Retrieved Context Preview:[/bold]")
        preview = context[:600] + "..." if len(context) > 600 else context
        self.console.print(
            Panel(preview, title="Context (truncated)", border_style="dim")
        )
        self.console.print()
    
    def display_response(self, answer: str):
        """Display the generated response."""
        self.console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Claude's Response[/bold green]",
                border_style="green",
            )
        )
    
    def display_sources(self, sources: List[Dict[str, str]]):
        """Display the sources used."""
        self.console.print("\n[bold]Sources Used:[/bold]")
        for i, source in enumerate(sources, 1):
            source_name = source["source"]
            score = source["score"]
            self.console.print(f"  {i}. [cyan]{source_name}[/cyan] (relevance: {score})")
        self.console.print()


def main():
    """Run generation as a module with rich console output."""
    logging.basicConfig(level=logging.ERROR)
    
    console = GenerationConsole()
    
    # Demo configuration
    DEMO_COLLECTION = "golden_demo"
    DEMO_PERSIST_PATH = "./data/demo_chroma_db"
    DEMO_QUERY = "How do I create a button with a loading state in the Golden design system?"
    
    # Phase header
    console.phase_header(
        phase_num=6,
        title="Response Generation",
        description=(
            "Generate contextual answers using Claude with retrieved documentation. "
            "This demonstrates the intersection of human expertise (documentation) "
            "and AI capabilities (understanding and synthesis)."
        )
    )
    
    # Initialize RAG generator
    with console.spinner("Initializing RAG generator"):
        with suppress_output():
            generator = RAGGenerator(
                collection_name=DEMO_COLLECTION,
                persist_path=DEMO_PERSIST_PATH,
                top_k=3
            )
            if not generator.initialize():
                console.error("Failed to initialize RAG generator")
                return
    
    # Check API key
    console.info("Checking API configuration...")
    if not generator.check_api_key():
        console.error("ANTHROPIC_API_KEY not set in environment")
        console.console.print(
            "[dim]Set the API key in .env file to run this demo[/dim]"
        )
        return
    
    # Check index status
    console.info("Checking index status...")
    index_status = generator.check_index_status()
    
    if not index_status["has_documents"]:
        console.warning("No documents indexed. Please run indexing first:")
        console.info("  python3 src/ingestion/indexing.py")
        return
    
    console.console.print(f"  Found [green]{index_status['document_count']}[/green] indexed documents")
    console.console.print()
    
    # Generate response
    console.console.print(f"\n[bold yellow]Query:[/bold yellow] {DEMO_QUERY}\n")
    
    with console.spinner("Generating response with Claude"):
        with suppress_output():
            response = generator.generate_response(
                query=DEMO_QUERY,
                include_context=True,
                max_context_docs=3
            )
    
    if "error" in response:
        console.error(f"Generation failed: {response['error']}")
        return
    
    # Display context preview
    if "context" in response:
        console.display_context_preview(response["context"])
    
    # Display the response
    console.display_response(response["answer"])
    
    # Display sources
    if "sources" in response:
        console.display_sources(response["sources"])
    
    console.success("Generation completed successfully!")


if __name__ == "__main__":
    main()
