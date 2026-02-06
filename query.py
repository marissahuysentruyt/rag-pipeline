#!/usr/bin/env python3
"""
CLI tool for querying design system documentation.

Usage:
    # Query with RAG (requires indexed documents)
    python query.py "How do I use a Button component?"

    # Query without RAG (direct LLM, no retrieval)
    python query.py --no-rag "How do I use a Button component?"

    # Interactive mode
    python query.py --interactive

    # Interactive mode without RAG
    python query.py --interactive --no-rag

    # With filters
    python query.py "Color guidelines" --domain spectrum.adobe.com

    # Adjust number of results
    python query.py "Button examples" --top-k 10
"""

import argparse
import logging
import os
import sys
from contextlib import contextmanager

from dotenv import load_dotenv

from demo.utils import DemoConsole
from src.generation import RAGGenerator, LLMConfig, ChatMessage, MessageRole
from src.generation.providers.anthropic import AnthropicProvider

DEFAULT_COLLECTION = "golden_demo"
DEFAULT_PERSIST_PATH = "./data/demo_chroma_db"
DEFAULT_LLM_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = (
    "You are an expert assistant helping developers use Golden design system "
    "components and guidelines."
)

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@contextmanager
def suppress_output():
    """Suppress stdout/stderr at the OS level for noisy C libraries."""
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


def run_no_rag_query(console: DemoConsole, llm: AnthropicProvider, question: str):
    """Send a query directly to Claude without any RAG context."""
    console.console.print(f"\n[bold yellow]Query:[/bold yellow] {question}\n")

    with console.spinner("Generating response (no RAG context)"):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        try:
            result = llm.chat(messages)
        except Exception as e:
            console.error(f"Generation failed: {e}")
            return

    console.markdown_panel(
        result.text,
        title="Claude's Response",
        border_style="red",
    )
    console.console.print(
        "[dim]This response was generated without retrieval-augmented context.\n"
        "The LLM is relying only on its general training data.[/dim]\n"
    )


def run_rag_query(
    console: DemoConsole,
    generator: RAGGenerator,
    question: str,
    show_sources: bool = True,
):
    """Run a query through the full RAG pipeline."""
    console.console.print(f"\n[bold yellow]Query:[/bold yellow] {question}\n")

    with console.spinner("Searching documentation and generating answer"):
        with suppress_output():
            response = generator.generate_response(
                query=question,
                include_context=True,
                max_context_docs=3,
            )

    if "error" in response:
        console.error(f"Generation failed: {response['error']}")
        return

    # Display context preview
    if "context" in response:
        context = response["context"]
        preview = context[:600] + "..." if len(context) > 600 else context
        console.sample_output("Retrieved Context", preview)

    # Display answer
    console.markdown_panel(
        response["answer"],
        title="Claude's Response",
        border_style="red",
    )

    # Display sources
    if show_sources and "sources" in response:
        sources = response["sources"]
        if sources:
            console.console.print("[bold]Sources Used:[/bold]")
            for i, source in enumerate(sources, 1):
                name = source["source"]
                score = source["score"]
                console.console.print(
                    f"  {i}. [cyan]{name}[/cyan] (relevance: {score})"
                )
            console.console.print()


def interactive_mode(
    console: DemoConsole,
    generator: RAGGenerator = None,
    llm: AnthropicProvider = None,
    no_rag: bool = False,
):
    """Run in interactive mode for multiple queries."""
    mode_label = "Direct LLM" if no_rag else "RAG"
    console.welcome_banner(
        title="Design System Documentation Search",
        subtitle=f"Mode: {mode_label} | Powered by Claude",
    )
    console.console.print(
        "[dim]Type your questions below. Type 'exit' or 'quit' to leave.[/dim]"
    )
    console.console.print("[dim]Type 'help' for available commands.[/dim]\n")

    show_sources = True

    while True:
        try:
            question = console.console.input("[bold cyan]> [/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                console.console.print("\n[dim]Goodbye![/dim]\n")
                break

            if question.lower() == "help":
                console.console.print("\n[bold]Available commands:[/bold]")
                console.console.print(
                    "  [cyan]exit, quit, q[/cyan]     - Exit the program"
                )
                console.console.print(
                    "  [cyan]help[/cyan]              - Show this help message"
                )
                console.console.print(
                    "  [cyan]sources on/off[/cyan]    - Toggle source display"
                )
                console.console.print(
                    "  [cyan]clear[/cyan]             - Clear the screen\n"
                )
                continue

            if question.lower().startswith("sources"):
                parts = question.lower().split()
                if len(parts) > 1:
                    if parts[1] == "off":
                        show_sources = False
                        console.console.print("[dim]Source display disabled[/dim]")
                    elif parts[1] == "on":
                        show_sources = True
                        console.console.print("[dim]Source display enabled[/dim]")
                continue

            if question.lower() == "clear":
                console.console.clear()
                continue

            if no_rag:
                run_no_rag_query(console, llm, question)
            else:
                run_rag_query(
                    console, generator, question, show_sources=show_sources
                )

        except KeyboardInterrupt:
            console.console.print("\n\n[dim]Goodbye![/dim]\n")
            break
        except EOFError:
            console.console.print("\n\n[dim]Goodbye![/dim]\n")
            break


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query design system documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query (RAG mode)
  python query.py "How do I use a Button component?"

  # Single query (no RAG — direct LLM)
  python query.py --no-rag "How do I use a Button component?"

  # Interactive mode
  python query.py --interactive

  # Interactive mode without RAG
  python query.py --interactive --no-rag

  # With domain filter
  python query.py "Color guidelines" --domain spectrum.adobe.com

  # Hide sources
  python query.py "What is Golden?" --no-sources
        """,
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG retrieval — send query directly to Claude",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    parser.add_argument(
        "--domain",
        help="Filter by domain (e.g., spectrum.adobe.com)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source citations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    load_dotenv()

    if not args.interactive and not args.question:
        parser.print_help()
        sys.exit(1)

    console = DemoConsole()

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.error("ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    if args.no_rag:
        # Direct LLM mode — no retrieval, no vector store needed
        console.console.print()
        console.warning("Running in no-RAG mode — no document retrieval")
        console.console.print(
            "[dim]Responses rely only on the LLM's general training data.[/dim]\n"
        )

        llm_config = LLMConfig(
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.3,
            max_tokens=2000,
            api_key=api_key,
        )
        llm = AnthropicProvider(llm_config)

        if args.interactive:
            interactive_mode(console, llm=llm, no_rag=True)
        else:
            run_no_rag_query(console, llm, args.question)

    else:
        # RAG mode
        try:
            with console.spinner("Loading RAG system"):
                with suppress_output():
                    generator = RAGGenerator(
                        collection_name=DEFAULT_COLLECTION,
                        persist_path=DEFAULT_PERSIST_PATH,
                        top_k=args.top_k,
                    )
                    if not generator.initialize():
                        console.error("Failed to initialize RAG system")
                        console.console.print(
                            "\n[dim]Make sure you have:\n"
                            "  1. Set ANTHROPIC_API_KEY in .env\n"
                            "  2. Run the indexer first: python -m src.ingestion.indexing[/dim]\n"
                        )
                        sys.exit(1)
        except Exception as e:
            console.error(f"Error initializing RAG system: {e}")
            console.console.print(
                "\n[dim]Make sure you have:\n"
                "  1. Set ANTHROPIC_API_KEY in .env\n"
                "  2. Run the indexer first: python -m src.ingestion.indexing[/dim]\n"
            )
            sys.exit(1)

        if args.interactive:
            interactive_mode(console, generator=generator)
        else:
            console.welcome_banner(
                title="Design System Documentation Search",
                subtitle="Powered by RAG + Claude",
            )
            run_rag_query(
                console,
                generator,
                args.question,
                show_sources=not args.no_sources,
            )


if __name__ == "__main__":
    main()
