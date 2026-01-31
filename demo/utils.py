"""
Shared utilities for RAG pipeline demos.

Provides consistent formatting and output using rich.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


class DemoConsole:
    """Wrapper around rich.Console for consistent demo formatting."""

    PHASE_COLORS = {
        1: "cyan",      # Ingest
        2: "green",     # Chunk
        3: "yellow",    # Embed
        4: "magenta",   # Index
        5: "blue",      # Retrieve
        6: "red",       # Generate
    }

    PHASE_NAMES = {
        1: "Document Ingestion",
        2: "Document Chunking",
        3: "Embedding Generation",
        4: "Vector Indexing",
        5: "Semantic Retrieval",
        6: "Response Generation",
    }

    def __init__(self):
        self.console = Console()

    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = self.PHASE_COLORS.get(phase_num, "white")
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

    def sample_output(self, title: str, content: str, syntax: Optional[str] = None, line_numbers: bool = False):
        """Display sample output in a panel."""
        if syntax:
            renderable = Syntax(content, syntax, theme="monokai", line_numbers=line_numbers)
        else:
            renderable = content
        self.console.print(Panel(
            renderable,
            title=f"[bold]Sample: {title}[/bold]",
            border_style="dim"
        ))
        self.console.print()

    def markdown_panel(self, content: str, title: str, border_style: str = "green"):
        """Display markdown content in a styled panel."""
        self.console.print(Panel(
            Markdown(content),
            title=f"[bold]{title}[/bold]",
            border_style=border_style
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

    def error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]{message}[/red]")

    def transition(self, message: str):
        """Display a transition message between phases."""
        self.console.print()
        self.console.rule(f"[bold dim]{message}[/bold dim]")
        self.console.print()

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

    def welcome_banner(self, title: str, subtitle: str, items: Optional[List[str]] = None):
        """Display a welcome banner."""
        content = f"[bold cyan]{title}[/bold cyan]\n\n[dim]{subtitle}[/dim]"
        if items:
            content += "\n\n" + "\n".join(f"  {i+1}. {item}" for i, item in enumerate(items))
        self.console.print()
        self.console.print(Panel.fit(
            content,
            border_style="cyan",
            padding=(1, 2)
        ))
        self.console.print()

    def completion_banner(self, title: str, message: str):
        """Display a completion banner."""
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold green]{title}[/bold green]\n\n{message}",
            border_style="green",
            padding=(1, 2)
        ))
        self.console.print()
