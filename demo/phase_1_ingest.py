#!/usr/bin/env python3
"""
Phase 1: Document Ingestion Demo

Demonstrates how the RAG pipeline loads and parses documentation files,
extracting content and metadata from markdown with YAML frontmatter.

Uses: src.ingestion.document_processor.DocumentProcessor
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.utils import DemoConsole
from src.ingestion.document_processor import DocumentProcessor


def run_demo(data_path: Path = None) -> dict:
    """
    Run the document ingestion demo.

    Args:
        data_path: Path to documentation files (defaults to data/golden/docs)

    Returns:
        Dict with ingestion results for next phase
    """
    console = DemoConsole()
    data_path = data_path or Path("data/golden/docs")

    # Phase header
    console.phase_header(
        phase_num=1,
        title="Document Ingestion",
        description=(
            "Load and parse markdown documentation files with YAML frontmatter. "
            "This phase extracts both content and structured metadata from each file."
        )
    )

    # Initialize the existing DocumentProcessor
    console.info("Initializing DocumentProcessor...")
    processor = DocumentProcessor(min_chunk_size=200, max_chunk_size=1500)

    # Find markdown files
    md_files = list(data_path.glob("*.md"))
    tsx_files = list(Path("data/golden/components").glob("*.tsx"))
    all_files = md_files + tsx_files

    console.list_files([f.name for f in md_files], "Markdown Documentation Files")
    console.list_files([f.name for f in tsx_files], "TypeScript Component Files")

    # Parse each markdown file
    all_docs = []
    for file_path in md_files:
        with console.spinner(f"Parsing {file_path.name}"):
            doc = processor.parse_markdown_file(file_path)
            if doc:
                all_docs.append({"path": file_path, "doc": doc})

    # Also read TypeScript files (raw content, no frontmatter)
    for file_path in tsx_files:
        with console.spinner(f"Reading {file_path.name}"):
            content = file_path.read_text()
            all_docs.append({
                "path": file_path,
                "doc": {
                    "metadata": {
                        "title": file_path.stem,
                        "component": file_path.stem,
                        "category": "Implementation",
                        "framework": "React/TypeScript",
                        "file_type": "tsx",
                        "domain": "golden.design"
                    },
                    "content": content
                }
            })

    # Summary statistics
    md_count = len([d for d in all_docs if d["path"].suffix == ".md"])
    tsx_count = len([d for d in all_docs if d["path"].suffix == ".tsx"])
    total_content = sum(len(d["doc"]["content"]) for d in all_docs)

    console.summary_table("Ingestion Summary", {
        "Markdown files parsed": md_count,
        "TypeScript files read": tsx_count,
        "Total documents": len(all_docs),
        "Total content size": f"{total_content:,} characters",
        "Avg document size": f"{total_content // len(all_docs):,} characters" if all_docs else "0",
    })

    # Sample output - show metadata from first markdown doc
    md_docs = [d for d in all_docs if d["path"].suffix == ".md"]
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

    return {"documents": all_docs, "processor": processor}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Document Ingestion Demo")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/golden/docs"),
        help="Path to documentation files"
    )
    args = parser.parse_args()

    run_demo(args.data_path)
