"""
Document processor for design system documentation.
Parses markdown files with frontmatter, extracts code examples, and creates chunks.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import yaml

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

        Returns:
            List of dicts with 'type' (code/text), 'content', and 'start_pos'
        """
        blocks = []
        last_pos = 0

        # Pattern to match [code]...[/code] blocks
        code_pattern = re.compile(r'\[code\](.*?)\[/code\]', re.DOTALL)

        for match in code_pattern.finditer(content):
            # Add text before code block
            if match.start() > last_pos:
                text_content = content[last_pos:match.start()].strip()
                if text_content:
                    blocks.append({
                        "type": "text",
                        "content": text_content,
                        "start_pos": last_pos
                    })

            # Add code block
            code_content = match.group(1).strip()
            if code_content:
                blocks.append({
                    "type": "code",
                    "content": code_content,
                    "start_pos": match.start()
                })

            last_pos = match.end()

        # Add remaining text
        if last_pos < len(content):
            text_content = content[last_pos:].strip()
            if text_content:
                blocks.append({
                    "type": "text",
                    "content": text_content,
                    "start_pos": last_pos
                })

        return blocks if blocks else [{"type": "text", "content": content, "start_pos": 0}]

    def extract_sections(self, content: str) -> List[Dict]:
        """
        Split content into sections based on markdown headings.

        Returns:
            List of dicts with 'heading', 'level', 'content'
        """
        sections = []
        lines = content.split('\n')
        current_section = {
            "heading": None,
            "level": 0,
            "content": []
        }

        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append({
                        **current_section,
                        "content": '\n'.join(current_section["content"]).strip()
                    })

                # Start new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                current_section = {
                    "heading": heading,
                    "level": level,
                    "content": []
                }
            else:
                current_section["content"].append(line)

        # Add final section
        if current_section["content"]:
            sections.append({
                **current_section,
                "content": '\n'.join(current_section["content"]).strip()
            })

        return sections

    def create_chunks(self, document: Dict) -> List[DocumentChunk]:
        """
        Create intelligent chunks from a parsed document.

        Strategy:
        - Split by headings to preserve context
        - Keep code blocks intact
        - Combine small sections
        - Split large sections while preserving code blocks

        Args:
            document: Dict with 'metadata' and 'content' keys

        Returns:
            List of DocumentChunk objects
        """
        metadata = document["metadata"]
        content = document["content"]
        chunks = []

        # Extract sections by heading
        sections = self.extract_sections(content)

        if not sections:
            # No headings, treat as single section
            sections = [{"heading": None, "level": 0, "content": content}]

        for section in sections:
            section_content = section["content"]
            section_heading = section["heading"]

            # Extract code blocks from this section
            blocks = self.extract_code_blocks(section_content)

            # Create chunks from blocks
            current_chunk_content = []
            current_chunk_has_code = False

            for block in blocks:
                block_content = block["content"]
                block_type = block["type"]

                # Add heading to first chunk of section
                if section_heading and not current_chunk_content:
                    if section["level"] <= 3:  # Only include h1-h3
                        current_chunk_content.append(f"{'#' * section['level']} {section_heading}\n")

                # Estimate current size
                current_size = sum(len(c) for c in current_chunk_content)
                block_size = len(block_content)

                # If adding this block would exceed max size, save current chunk
                if current_chunk_content and current_size + block_size > self.max_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk_content).strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk_type = "mixed" if current_chunk_has_code else "text"
                        chunks.append(DocumentChunk(
                            content=chunk_text,
                            metadata=metadata,
                            chunk_type=chunk_type,
                            heading=section_heading
                        ))
                    current_chunk_content = []
                    current_chunk_has_code = False

                    # Re-add heading to new chunk
                    if section_heading and section["level"] <= 3:
                        current_chunk_content.append(f"{'#' * section['level']} {section_heading}\n")

                # Add block to current chunk
                if block_type == "code":
                    current_chunk_content.append(f"```\n{block_content}\n```")
                    current_chunk_has_code = True
                else:
                    current_chunk_content.append(block_content)

            # Save final chunk from this section
            if current_chunk_content:
                chunk_text = '\n\n'.join(current_chunk_content).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_type = "mixed" if current_chunk_has_code else "text"
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata=metadata,
                        chunk_type=chunk_type,
                        heading=section_heading
                    ))

        # If no chunks were created (all content too short), create one chunk
        if not chunks and len(content.strip()) >= self.min_chunk_size:
            blocks = self.extract_code_blocks(content)
            has_code = any(b["type"] == "code" for b in blocks)
            chunks.append(DocumentChunk(
                content=content,
                metadata=metadata,
                chunk_type="mixed" if has_code else "text",
                heading=None
            ))

        logger.debug(f"Created {len(chunks)} chunks from document: {metadata.get('title', 'Unknown')}")
        return chunks

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
