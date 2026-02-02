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

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

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

    # Import phase modules
    import subprocess
    import sys
    
    def run_phase_module(module_name, description):
        """Run a phase module and display output."""
        console.transition(f"Starting {description}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", module_name],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT
            )
            if result.returncode == 0:
                # Print the output directly to maintain formatting
                console.console.print(result.stdout)
            else:
                console.error(f"Phase failed: {result.stderr}")
                return None
        except Exception as e:
            console.error(f"Error running phase: {e}")
            return None
        return True

    # Run phases using modules
    success_count = 0
    
    # Phase 1: Ingestion
    if run_phase_module("src.ingestion", "Phase 1: Document Ingestion"):
        success_count += 1
    time.sleep(0.5)

    # Phase 2: Chunking
    if run_phase_module("src.processing", "Phase 2: Semantic Chunking"):
        success_count += 1
    time.sleep(0.5)

    # Phase 3: Embedding
    if run_phase_module("src.embedding", "Phase 3: Embedding Generation"):
        success_count += 1
    time.sleep(0.5)

    # Phase 4: Indexing
    if run_phase_module("src.ingestion.indexing", "Phase 4: Vector Indexing"):
        success_count += 1
    time.sleep(0.5)

    # Phase 5: Retrieval
    if run_phase_module("src.retrieval", "Phase 5: Semantic Retrieval"):
        success_count += 1
    time.sleep(0.5)

    # Phase 6: Generation (optional)
    generation_success = False
    if not skip_generation:
        if run_phase_module("src.generation", "Phase 6: AI Response Generation"):
            generation_success = True
            success_count += 1
    else:
        console.console.print(
            "\n[yellow]Skipping Phase 6 (--skip-generation flag)[/yellow]\n"
        )

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
    console.key_value("Phases completed", f"{success_count}/6")
    if not skip_generation:
        console.key_value("Generation phase", "Completed" if generation_success else "Skipped")
    else:
        console.key_value("Generation phase", "Skipped by user")
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
