#!/usr/bin/env python3
"""
Full RAG Pipeline Demo

Runs all phases in sequence with clear transitions,
demonstrating the complete journey from raw documentation to AI-generated answers.

This script showcases the intersection of human expertise (carefully authored
documentation) and AI capabilities (semantic understanding and synthesis).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from demo.utils import DemoConsole


def run_full_demo(skip_generation: bool = False):
    """
    Run the complete RAG pipeline demo.

    Args:
        skip_generation: If True, skip Phase 6 (useful if no API key)
    """
    console = DemoConsole()

    # Welcome banner
    console.welcome_banner(
        title="RAG Pipeline Demo",
        subtitle="The Intersection of Human Expertise and AI",
        items=[
            "Document Ingestion",
            "Semantic Chunking",
            "Embedding Generation",
            "Vector Indexing",
            "Semantic Retrieval",
            "AI Response Generation",
        ],
    )

    console.console.print(
        "[dim]This demo walks through all 6 phases of a RAG pipeline, "
        "showing how documentation is transformed into an intelligent "
        "question-answering system.[/dim]\n"
    )

    # Pause for effect
    time.sleep(1)

    # Import phase demos
    from demo.phase_1_ingest import run_demo as phase_1
    from demo.phase_2_chunk import run_demo as phase_2
    from demo.phase_3_embed import run_demo as phase_3
    from demo.phase_4_index import run_demo as phase_4
    from demo.phase_5_retrieve import run_demo as phase_5
    from demo.phase_6_generate import run_demo as phase_6

    # Phase 1: Ingestion
    console.transition("Starting Phase 1: Document Ingestion")
    result_1 = phase_1()
    time.sleep(0.5)

    # Phase 2: Chunking
    console.transition("Starting Phase 2: Semantic Chunking")
    result_2 = phase_2(documents=result_1["documents"])
    time.sleep(0.5)

    # Phase 3: Embedding (sample only to save time)
    console.transition("Starting Phase 3: Embedding Generation")
    result_3 = phase_3(chunks=result_2["chunks"][:5])
    time.sleep(0.5)

    # Phase 4: Indexing
    console.transition("Starting Phase 4: Vector Indexing")
    result_4 = phase_4(chunks=result_2["chunks"])
    time.sleep(0.5)

    # Phase 5: Retrieval
    console.transition("Starting Phase 5: Semantic Retrieval")
    result_5 = phase_5(ensure_indexed=False)
    time.sleep(0.5)

    # Phase 6: Generation (optional)
    if not skip_generation:
        console.transition("Starting Phase 6: AI Response Generation")
        try:
            result_6 = phase_6()
        except Exception as e:
            console.error(f"Phase 6 failed: {e}")
            console.console.print(
                "[dim]Ensure ANTHROPIC_API_KEY is set in .env[/dim]"
            )
            result_6 = None
    else:
        console.console.print(
            "\n[yellow]Skipping Phase 6 (--skip-generation flag)[/yellow]\n"
        )
        result_6 = None

    # Final summary
    console.completion_banner(
        title="Demo Complete!",
        message=(
            "You've seen how the RAG pipeline transforms raw documentation into\n"
            "intelligent, contextual responses by combining:\n\n"
            "[cyan]Human Expertise:[/cyan] Carefully authored documentation\n"
            "[yellow]AI Capabilities:[/yellow] Semantic understanding and synthesis\n\n"
            "The result is an AI assistant that provides accurate, sourced answers\n"
            "grounded in your organization's knowledge."
        ),
    )

    # Print summary stats
    console.console.print("[bold]Pipeline Summary:[/bold]")
    console.key_value("Documents ingested", len(result_1["documents"]))
    console.key_value("Chunks created", len(result_2["chunks"]))
    console.key_value("Chunks indexed", result_4["indexed_count"])
    console.key_value(
        "Sample queries run",
        len(result_5["results"]) if result_5 else 0,
    )
    if result_6 and "answer" in result_6:
        console.key_value("AI response generated", "Yes")
    console.console.print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full RAG pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full demo (requires ANTHROPIC_API_KEY)
  python demo/run_full_pipeline.py

  # Skip AI generation phase (no API key needed)
  python demo/run_full_pipeline.py --skip-generation
        """,
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip Phase 6 (AI generation) - useful if no API key",
    )
    args = parser.parse_args()

    run_full_demo(skip_generation=args.skip_generation)


if __name__ == "__main__":
    main()
