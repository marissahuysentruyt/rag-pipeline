"""
Base interface for document chunking strategies.

This module defines the abstract base class for document chunkers,
allowing the system to support multiple chunking strategies
(fixed-size, semantic, markdown-aware, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ChunkType(Enum):
    """Types of chunks."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"
    QUOTE = "quote"


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.

    Attributes:
        content: The text content of the chunk
        chunk_type: Type of chunk (text, code, etc.)
        metadata: Original document metadata plus chunk-specific data
        start_index: Character index where chunk starts in original document
        end_index: Character index where chunk ends in original document
        heading: Section heading this chunk belongs to (if applicable)
        chunk_id: Unique identifier for this chunk
    """
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: Optional[Dict[str, Any]] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    heading: Optional[str] = None
    chunk_id: Optional[str] = None

    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if self.chunk_id is None and self.metadata:
            # Generate ID from source + position
            source_id = self.metadata.get('source_id', 'unknown')
            start = self.start_index or 0
            self.chunk_id = f"{source_id}_{start}"


@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking.

    Attributes:
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        preserve_code_blocks: Whether to keep code blocks intact
        preserve_tables: Whether to keep tables intact
        respect_sentence_boundaries: Split on sentence boundaries when possible
        additional_params: Strategy-specific parameters
    """
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    chunk_overlap: int = 100
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    respect_sentence_boundaries: bool = True
    additional_params: Optional[Dict[str, Any]] = None


class ChunkerStrategy(ABC):
    """
    Abstract base class for document chunking strategies.

    Each strategy is responsible for:
    1. Splitting documents into appropriate chunks
    2. Preserving important structure (code, tables, etc.)
    3. Maintaining context through overlap or metadata
    4. Respecting size constraints
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize the chunking strategy.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk a single text document.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        pass

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents, each with 'content' and 'metadata' keys

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        return all_chunks

    def get_chunk_count(self, text: str) -> int:
        """
        Estimate how many chunks will be created from text.

        Args:
            text: Input text

        Returns:
            Estimated number of chunks
        """
        # Default simple estimation
        avg_chunk_size = (self.config.min_chunk_size + self.config.max_chunk_size) // 2
        return max(1, len(text) // avg_chunk_size)

    def validate_chunk(self, chunk: Chunk) -> bool:
        """
        Validate that a chunk meets size requirements.

        Args:
            chunk: Chunk to validate

        Returns:
            True if chunk is valid, False otherwise
        """
        length = len(chunk.content)

        # Allow chunks smaller than min if they're special types
        if chunk.chunk_type in [ChunkType.CODE, ChunkType.TABLE]:
            return length <= self.config.max_chunk_size

        return (self.config.min_chunk_size <= length <= self.config.max_chunk_size)

    def merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge consecutive small chunks if possible.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with small ones merged
        """
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Try to merge if both are text and combined size is acceptable
            if (current.chunk_type == ChunkType.TEXT and
                next_chunk.chunk_type == ChunkType.TEXT and
                len(current.content) + len(next_chunk.content) <= self.config.max_chunk_size):

                # Merge
                current = Chunk(
                    content=current.content + "\n\n" + next_chunk.content,
                    chunk_type=ChunkType.TEXT,
                    metadata=current.metadata,
                    start_index=current.start_index,
                    end_index=next_chunk.end_index,
                    heading=current.heading
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the chunking strategy.

        Returns:
            Dictionary with strategy metadata
        """
        return {
            "strategy": self.__class__.__name__,
            "min_chunk_size": self.config.min_chunk_size,
            "max_chunk_size": self.config.max_chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "preserve_code_blocks": self.config.preserve_code_blocks,
            "preserve_tables": self.config.preserve_tables
        }


class ChunkingError(Exception):
    """Base exception for chunking errors."""
    pass


class InvalidChunkSizeError(ChunkingError):
    """Raised when chunk size configuration is invalid."""
    pass
