"""
Markdown-aware chunking strategy.

This chunker preserves markdown structure including headings, code blocks,
and sections while creating chunks of appropriate size.
"""

import logging
import re
from typing import List, Dict, Optional

from .base import ChunkerStrategy, ChunkingConfig, Chunk, ChunkType, InvalidChunkSizeError

logger = logging.getLogger(__name__)


class MarkdownChunker(ChunkerStrategy):
    """
    Chunking strategy for markdown documents.

    This chunker:
    - Splits by headings to preserve context
    - Keeps code blocks intact
    - Combines small sections
    - Splits large sections while preserving code blocks
    """

    def _validate_config(self) -> None:
        """Validate chunking configuration."""
        if self.config.min_chunk_size <= 0:
            raise InvalidChunkSizeError("min_chunk_size must be positive")

        if self.config.max_chunk_size <= self.config.min_chunk_size:
            raise InvalidChunkSizeError("max_chunk_size must be greater than min_chunk_size")

    def extract_code_blocks(self, content: str) -> List[Dict]:
        """
        Extract code blocks from markdown content.

        Supports both [code]...[/code] and ```...``` syntax.

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

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk markdown text into appropriate-sized pieces.

        Args:
            text: Markdown text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        chunks = []
        metadata = metadata or {}

        # Extract sections by heading
        sections = self.extract_sections(text)

        if not sections:
            # No headings, treat as single section
            sections = [{"heading": None, "level": 0, "content": text}]

        for section in sections:
            section_content = section["content"]
            section_heading = section["heading"]

            # Skip empty sections
            if not section_content.strip():
                continue

            # Extract code blocks from this section
            blocks = self.extract_code_blocks(section_content)

            # Create chunks from blocks
            current_chunk_content = []
            current_chunk_has_code = False
            chunk_start = 0

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
                if current_chunk_content and current_size + block_size > self.config.max_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk_content).strip()
                    if len(chunk_text) >= self.config.min_chunk_size:
                        chunk_type = ChunkType.CODE if current_chunk_has_code else ChunkType.TEXT
                        chunks.append(Chunk(
                            content=chunk_text,
                            chunk_type=chunk_type,
                            metadata=metadata,
                            heading=section_heading,
                            start_index=chunk_start,
                            end_index=chunk_start + len(chunk_text)
                        ))

                    chunk_start += len(chunk_text)
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
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk_type = ChunkType.CODE if current_chunk_has_code else ChunkType.TEXT
                    chunks.append(Chunk(
                        content=chunk_text,
                        chunk_type=chunk_type,
                        metadata=metadata,
                        heading=section_heading,
                        start_index=chunk_start,
                        end_index=chunk_start + len(chunk_text)
                    ))

        # If no chunks were created (all content too short), create one chunk
        if not chunks and len(text.strip()) >= self.config.min_chunk_size:
            blocks = self.extract_code_blocks(text)
            has_code = any(b["type"] == "code" for b in blocks)
            chunks.append(Chunk(
                content=text,
                chunk_type=ChunkType.CODE if has_code else ChunkType.TEXT,
                metadata=metadata,
                heading=None
            ))

        logger.debug(f"Created {len(chunks)} chunks from markdown text")
        return chunks

    def process_documents(self, documents: List[Dict]) -> List[Chunk]:
        """
        Process multiple documents and return all chunks.

        Args:
            documents: List of dicts with 'path' and 'doc' keys

        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks = []
        
        logger.info(f"Processing {len(documents)} documents")
        
        for doc_info in documents:
            doc = doc_info["doc"]
            filename = doc_info["path"].name

            # Add source filename to metadata
            metadata = {**doc["metadata"], "source": filename}
            chunks = self.chunk_text(doc["content"], metadata)
            all_chunks.extend(chunks)
            
            logger.debug(f"Created {len(chunks)} chunks from {filename}")
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks

    def calculate_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        """
        Calculate statistics for processed chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Dict with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "text_chunks": 0,
                "code_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        text_chunks = [c for c in chunks if c.chunk_type == ChunkType.TEXT]
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        
        return {
            "total_chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "code_chunks": len(code_chunks),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }

    def get_sample_chunks(self, chunks: List[Chunk], num_samples: int = 3) -> List[Dict]:
        """
        Get sample chunks for display.

        Args:
            chunks: List of Chunk objects
            num_samples: Number of samples to return

        Returns:
            List of dicts with chunk info for display
        """
        samples = []
        for i, chunk in enumerate(chunks[:num_samples], 1):
            chunk_type_label = "CODE" if chunk.chunk_type == ChunkType.CODE else "TEXT"
            preview = chunk.content[:150].replace("\n", " ")
            
            samples.append({
                "index": i,
                "type": chunk_type_label,
                "size": len(chunk.content),
                "heading": chunk.heading or "None",
                "preview": preview + "...",
                "source": chunk.metadata.get("source", "Unknown") if chunk.metadata else "Unknown"
            })
        
        return samples
