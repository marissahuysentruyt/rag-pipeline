#!/usr/bin/env python3
"""
Phase 6: Response Generation Demo

Demonstrates how Claude generates contextual answers using retrieved documentation.
This is where human expertise (documentation) meets AI capabilities.

Uses: src.query.rag_pipeline.DesignSystemRAG
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich.markdown import Markdown
from rich.panel import Panel

from demo.utils import DemoConsole
from demo.phase_4_index import DEMO_COLLECTION, DEMO_PERSIST_PATH

# Demo query
DEMO_QUERY = "How do I create a button with a loading state in the Golden design system?"


def run_demo(query: str = None, retrieval_results: list = None) -> dict:
    """
    Run the generation demo.

    Args:
        query: Custom query to use (defaults to DEMO_QUERY)
        retrieval_results: Pre-fetched retrieval results (optional)

    Returns:
        Dict with generation results
    """
    console = DemoConsole()

    console.phase_header(
        phase_num=6,
        title="Response Generation",
        description=(
            "Generate contextual answers using Claude with retrieved documentation. "
            "This demonstrates the intersection of human expertise (documentation) "
            "and AI capabilities (understanding and synthesis)."
        )
    )

    query = query or DEMO_QUERY

    # Get retrieval results if not provided
    if retrieval_results is None:
        console.info("Running retrieval for query...")
        from demo.phase_5_retrieve import run_demo as retrieve_demo

        retrieve_result = retrieve_demo(ensure_indexed=True)
        # Use the first result set
        if retrieve_result["results"]:
            retrieval_results = retrieve_result["results"][0]["documents"]
        else:
            retrieval_results = []
        console.transition("Continuing to generation")

    # Build context from retrieved documents
    console.info("Building context from retrieved documents...")
    context_parts = []
    for i, doc in enumerate(retrieval_results[:3], 1):
        source = doc.meta.get("component", doc.meta.get("title", "Unknown"))
        heading = doc.meta.get("heading", "")
        header = f"Source: {source}"
        if heading:
            header += f" > {heading}"
        context_parts.append(f"--- {header} ---\n{doc.content}")

    context = "\n\n".join(context_parts)

    # Show context preview
    console.console.print("[bold]Retrieved Context Preview:[/bold]")
    preview = context[:600] + "..." if len(context) > 600 else context
    console.console.print(
        Panel(preview, title="Context (truncated)", border_style="dim")
    )
    console.console.print()

    # Check for API key
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.error("ANTHROPIC_API_KEY not set in environment")
        console.console.print(
            "[dim]Set the API key in .env file to run this demo[/dim]"
        )
        return {"error": "API key not configured"}

    # Initialize Claude via existing DesignSystemRAG
    console.info("Initializing Claude...")

    from src.query.rag_pipeline import DesignSystemRAG

    try:
        rag = DesignSystemRAG(
            collection_name=DEMO_COLLECTION,
            persist_path=DEMO_PERSIST_PATH,
            top_k=3,
        )
    except ValueError as e:
        console.error(f"Failed to initialize RAG: {e}")
        return {"error": str(e)}

    # Generate response
    console.console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    with console.spinner("Generating response with Claude"):
        result = rag.query(query)

    # Display the response
    console.console.print(
        Panel(
            Markdown(result["answer"]),
            title="[bold green]Claude's Response[/bold green]",
            border_style="green",
        )
    )

    # Show sources used
    console.console.print("\n[bold]Sources Used:[/bold]")
    for i, doc in enumerate(result["documents"], 1):
        source = doc.meta.get("component", doc.meta.get("title", "Unknown"))
        score = f"{doc.score:.3f}" if doc.score else "N/A"
        console.console.print(f"  {i}. [cyan]{source}[/cyan] (relevance: {score})")

    console.console.print()

    # Summary stats
    console.summary_table("Generation Stats", {
        "Model": result["metadata"]["model"],
        "Documents retrieved": result["metadata"]["num_documents"],
        "Query": query[:50] + "..." if len(query) > 50 else query,
    })

    return {
        "query": query,
        "answer": result["answer"],
        "documents": result["documents"],
        "metadata": result["metadata"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6: Response Generation Demo")
    parser.add_argument(
        "--query",
        type=str,
        default=DEMO_QUERY,
        help="Query to ask",
    )
    args = parser.parse_args()

    run_demo(query=args.query)
