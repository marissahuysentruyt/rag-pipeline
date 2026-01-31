"""
Main entry point for retrieval module.

Run with: python -m src.retrieval
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

from src.retrieval import ChromaRetriever, RetrievalConfig
from src.embedding.providers import SentenceTransformersProvider, EmbeddingConfig
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


class RetrievalConsole:
    """Simplified console output for retrieval module."""
    
    def __init__(self):
        self.console = Console()
    
    def phase_header(self, phase_num: int, title: str, description: str):
        """Display a prominent phase header."""
        color = "blue"
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
    
    def display_query_results(self, query: str, table_rows: List[tuple]):
        """Display results for a single query."""
        self.console.print(f"\n[bold yellow]Query:[/bold yellow] {query}")
        
        if not table_rows:
            self.console.print("[red]No results found[/red]")
            return
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Score", width=8)
        table.add_column("Component", width=15)
        table.add_column("Heading", width=20)
        table.add_column("Preview", width=40)
        
        for row in table_rows:
            table.add_row(*row)
        
        self.console.print(table)
        self.console.print()


def main():
    """Run retrieval as a module with rich console output."""
    logging.basicConfig(level=logging.INFO)
    
    console = RetrievalConsole()
    
    # Demo configuration
    DEMO_COLLECTION = "golden_demo"
    DEMO_PERSIST_PATH = "./data/demo_chroma_db"
    
    # Sample queries to demonstrate retrieval
    SAMPLE_QUERIES = [
        "How do I create a primary button?",
        "What props does the Card component accept?",
        "Show me how to use Modal with focus management",
    ]
    
    # Phase header
    console.phase_header(
        phase_num=5,
        title="Semantic Retrieval",
        description=(
            "Search for relevant documentation using semantic similarity. "
            "Queries are embedded and matched against indexed chunks to find "
            "the most relevant content."
        )
    )
    
    # Connect to vector store
    console.info("Connecting to vector store...")
    document_store = ChromaDocumentStore(
        collection_name=DEMO_COLLECTION,
        persist_path=DEMO_PERSIST_PATH
    )
    
    # Check index status
    console.info("Checking index status...")
    
    # Initialize embedder for queries first
    console.info("Initializing query embedder...")
    embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384
    )
    embedding_provider = SentenceTransformersProvider(embedding_config)
    
    with console.spinner("Loading embedder"):
        embedding_provider.load_model()
    
    # Now create retriever with embedding provider
    retriever = ChromaRetriever(
        config=RetrievalConfig(top_k=3),
        document_store=document_store,
        embedding_provider=embedding_provider
    )
    
    index_status = retriever.check_index_status()
    
    if not index_status["has_documents"]:
        console.warning("No documents indexed. Please run indexing first:")
        console.info("  python3 src/ingestion/indexing.py")
        return
    
    console.console.print(f"  Found [green]{index_status['document_count']}[/green] indexed documents")
    console.console.print()
    
    console.summary_table("Retrieval Configuration", {
        "Total indexed documents": index_status["document_count"],
        "Top K results": 3,
        "Embedding model": "all-MiniLM-L6-v2",
        "Collection": DEMO_COLLECTION,
    })
    
    # Process sample queries
    console.info(f"Processing {len(SAMPLE_QUERIES)} sample queries...")
    
    with console.spinner("Processing queries"):
        results = retriever.process_sample_queries(SAMPLE_QUERIES)
        table_rows = retriever.format_results_for_display(results)
    
    # Display results
    for i, (query_result, formatted_rows) in enumerate(zip(results, table_rows)):
        console.console.print(
            f"[green]Found {query_result['results_count']} relevant chunks[/green]"
        )
        console.display_query_results(query_result["query"], formatted_rows)
    
    console.success("Retrieval completed successfully!")


if __name__ == "__main__":
    main()
