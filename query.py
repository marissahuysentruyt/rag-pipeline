#!/usr/bin/env python3
"""
CLI tool for querying design system documentation using RAG.

Usage:
    # Single query
    python query.py "How do I use a Button component?"

    # Interactive mode
    python query.py --interactive

    # With filters
    python query.py "Color guidelines" --domain spectrum.adobe.com

    # Adjust number of results
    python query.py "Button examples" --top-k 10
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.query.rag_pipeline import DesignSystemRAG

DEFAULT_COLLECTION = "golden_demo"
DEFAULT_PERSIST_PATH = "./data/demo_chroma_db"

# Set up console for rich output
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def print_header():
    """Print welcome header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Design System Documentation Search[/bold cyan]\n"
        "Powered by RAG + Claude 4.5",
        border_style="cyan"
    ))
    console.print()


def print_sources(documents):
    """Print retrieved sources in a table."""
    if not documents:
        return

    table = Table(title="Sources", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan")
    table.add_column("URL", style="blue")
    table.add_column("Score", justify="right", style="green")

    for i, doc in enumerate(documents, 1):
        title = doc.meta.get('title', 'Unknown')
        url = doc.meta.get('url', 'N/A')
        score = f"{doc.score:.3f}" if doc.score else "N/A"

        # Truncate long titles
        if len(title) > 50:
            title = title[:47] + "..."

        table.add_row(str(i), title, url, score)

    console.print()
    console.print(table)


def format_answer(answer: str) -> str:
    """Format the answer for display."""
    # Clean up the answer if needed
    return answer.strip()


def run_query(rag: DesignSystemRAG, question: str, filters: dict = None, show_sources: bool = True):
    """Run a single query and display results."""
    console.print(f"\n[bold yellow]Question:[/bold yellow] {question}")
    console.print()

    with console.status("[bold green]Searching documentation and generating answer...", spinner="dots"):
        try:
            result = rag.query(question, filters=filters)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return

    # Display answer
    answer = format_answer(result['answer'])
    console.print(Panel(Markdown(answer), title="[bold green]Answer", border_style="green"))

    # Display sources if requested
    if show_sources and result['documents']:
        print_sources(result['documents'])

    console.print()


def interactive_mode(rag: DesignSystemRAG, filters: dict = None):
    """Run in interactive mode for multiple queries."""
    print_header()
    console.print("[dim]Type your questions below. Type 'exit' or 'quit' to leave.[/dim]")
    console.print("[dim]Type 'help' for available commands.[/dim]")
    console.print()

    show_sources = True

    while True:
        try:
            # Get user input
            question = console.input("[bold cyan]❯[/bold cyan] ").strip()

            if not question:
                continue

            # Handle commands
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("\n[dim]Goodbye![/dim]\n")
                break

            if question.lower() == 'help':
                console.print("\n[bold]Available commands:[/bold]")
                console.print("  [cyan]exit, quit, q[/cyan]     - Exit the program")
                console.print("  [cyan]help[/cyan]              - Show this help message")
                console.print("  [cyan]sources on/off[/cyan]    - Toggle source display")
                console.print("  [cyan]clear[/cyan]             - Clear the screen")
                console.print()
                continue

            if question.lower().startswith('sources'):
                parts = question.lower().split()
                if len(parts) > 1:
                    if parts[1] == 'off':
                        show_sources = False
                        console.print("[dim]Source display disabled[/dim]")
                    elif parts[1] == 'on':
                        show_sources = True
                        console.print("[dim]Source display enabled[/dim]")
                continue

            if question.lower() == 'clear':
                console.clear()
                print_header()
                continue

            # Run the query
            run_query(rag, question, filters=filters, show_sources=show_sources)

        except KeyboardInterrupt:
            console.print("\n\n[dim]Goodbye![/dim]\n")
            break
        except EOFError:
            console.print("\n\n[dim]Goodbye![/dim]\n")
            break


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query design system documentation using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python query.py "How do I use a Button component?"

  # Interactive mode
  python query.py --interactive

  # With domain filter
  python query.py "Color guidelines" --domain spectrum.adobe.com

  # Adjust number of results
  python query.py "Button examples" --top-k 10

  # Hide sources
  python query.py "What is Spectrum?" --no-sources
        """
    )

    parser.add_argument(
        'question',
        nargs='?',
        help='Question to ask (omit for interactive mode)'
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )

    parser.add_argument(
        '--domain',
        help='Filter by domain (e.g., spectrum.adobe.com)'
    )

    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source citations'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Load environment variables
    load_dotenv()

    # Validate input
    if not args.interactive and not args.question:
        parser.print_help()
        sys.exit(1)

    # Build filters (Chroma format)
    filters = None
    if args.domain:
        filters = {
            "field": "domain",
            "operator": "==",
            "value": args.domain
        }

    # Initialize RAG system
    try:
        with console.status("[bold green]Loading RAG system...", spinner="dots"):
            rag = DesignSystemRAG(
                collection_name=DEFAULT_COLLECTION,
                persist_path=DEFAULT_PERSIST_PATH,
                top_k=args.top_k
            )
    except Exception as e:
        console.print(f"[bold red]Error initializing RAG system:[/bold red] {e}")
        console.print("\n[dim]Make sure you have:")
        console.print("  1. Set ANTHROPIC_API_KEY in .env")
        console.print("  2. Run the indexer first: python src/ingestion/document_indexer.py")
        console.print("[/dim]\n")
        sys.exit(1)

    # Run in appropriate mode
    if args.interactive or not args.question:
        interactive_mode(rag, filters=filters)
    else:
        print_header()
        run_query(rag, args.question, filters=filters, show_sources=not args.no_sources)


if __name__ == "__main__":
    main()
