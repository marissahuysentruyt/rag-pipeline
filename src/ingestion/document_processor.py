"""
Document processor for design system documentation.
Parses markdown files with frontmatter, extracts code examples, and creates chunks.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from src.processing.chunkers import MarkdownChunker, ChunkingConfig, Chunk, ChunkType

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of documentation with metadata."""

    def __init__(
        self,
        content: str,
        metadata: Dict,
        chunk_type: str = "text",
        heading: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata
        self.chunk_type = chunk_type  # text, code, mixed
        self.heading = heading

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for indexing."""
        return {
            "content": self.content,
            "metadata": {
                **self.metadata,
                "chunk_type": self.chunk_type,
                "heading": self.heading
            }
        }

    def __repr__(self) -> str:
        return f"DocumentChunk(type={self.chunk_type}, heading={self.heading}, length={len(self.content)})"


class DocumentProcessor:
    """Processes crawled markdown files for indexing."""

    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1500):
        """
        Initialize document processor.

        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Initialize markdown chunker with configuration
        self.chunker = MarkdownChunker(
            ChunkingConfig(
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                preserve_code_blocks=True
            )
        )

    def parse_markdown_file(self, file_path: Path) -> Optional[Dict]:
        """
        Parse a markdown file with YAML frontmatter.

        Returns:
            Dict with 'metadata' and 'content' keys, or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split frontmatter and content
            if not content.startswith('---'):
                logger.warning(f"No frontmatter found in {file_path}")
                return None

            # Find the end of frontmatter
            parts = content.split('---', 2)
            if len(parts) < 3:
                logger.warning(f"Invalid frontmatter format in {file_path}")
                return None

            frontmatter = parts[1]
            markdown_content = parts[2].strip()

            # Parse YAML frontmatter
            metadata = yaml.safe_load(frontmatter)

            return {
                "metadata": metadata,
                "content": markdown_content
            }

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def extract_code_blocks(self, content: str) -> List[Dict]:
        """
        Extract code blocks from markdown content.

        Delegates to MarkdownChunker for consistency.

        Returns:
            List of dicts with 'type' (code/text), 'content', and 'start_pos'
        """
        return self.chunker.extract_code_blocks(content)

    def extract_sections(self, content: str) -> List[Dict]:
        """
        Split content into sections based on markdown headings.

        Delegates to MarkdownChunker for consistency.

        Returns:
            List of dicts with 'heading', 'level', 'content'
        """
        return self.chunker.extract_sections(content)

    def create_chunks(self, document: Dict) -> List[DocumentChunk]:
        """
        Create intelligent chunks from a parsed document.

        Delegates to MarkdownChunker and converts results to DocumentChunk format.

        Args:
            document: Dict with 'metadata' and 'content' keys

        Returns:
            List of DocumentChunk objects
        """
        metadata = document["metadata"]
        content = document["content"]

        # Use the chunker to create chunks
        chunks = self.chunker.chunk_text(content, metadata)

        # Convert from Chunk to DocumentChunk for backward compatibility
        document_chunks = []
        for chunk in chunks:
            # Map ChunkType to string
            chunk_type_str = "text"
            if chunk.chunk_type == ChunkType.CODE:
                chunk_type_str = "mixed"  # Keep "mixed" for backward compatibility
            elif chunk.chunk_type == ChunkType.TEXT:
                chunk_type_str = "text"

            document_chunks.append(DocumentChunk(
                content=chunk.content,
                metadata=metadata,
                chunk_type=chunk_type_str,
                heading=chunk.heading
            ))

        logger.debug(f"Created {len(document_chunks)} chunks from document: {metadata.get('title', 'Unknown')}")
        return document_chunks

    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Process a single markdown file and return chunks.

        Args:
            file_path: Path to markdown file

        Returns:
            List of DocumentChunk objects
        """
        document = self.parse_markdown_file(file_path)
        if not document:
            return []

        return self.create_chunks(document)

    def process_directory(self, directory: Path) -> List[DocumentChunk]:
        """
        Process all markdown files in a directory.

        Args:
            directory: Path to directory containing markdown files

        Returns:
            List of all DocumentChunk objects from all files
        """
        all_chunks = []
        md_files = list(directory.glob("*.md"))

        logger.info(f"Processing {len(md_files)} markdown files from {directory}")

        for file_path in md_files:
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(md_files)} files")
        return all_chunks


def main():
    """Test the document processor."""
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor()
    chunks = processor.process_directory(Path("data/raw/crawled"))

    print(f"\nProcessed {len(chunks)} chunks")
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Type: {chunk.chunk_type}")
        print(f"Heading: {chunk.heading}")
        print(f"Length: {len(chunk.content)} chars")
        print(f"Content preview: {chunk.content[:200]}...")


if __name__ == "__main__":
    main()
