"""
Code-aware chunking strategy.

This chunker is designed for source code files, preserving:
- Complete function/method definitions
- Complete class definitions
- Docstrings and comments
- Import statements
"""

import logging
import re
from typing import List, Dict, Any, Optional

from .base import ChunkerStrategy, ChunkingConfig, Chunk, ChunkType, InvalidChunkSizeError

logger = logging.getLogger(__name__)


class CodeChunker(ChunkerStrategy):
    """
    Chunking strategy for source code files.
    
    This chunker:
    - Extracts complete functions/classes as individual chunks
    - Preserves docstrings and comments
    - Keeps import statements with the first chunk
    - Handles multiple programming languages
    """
    
    def _validate_config(self) -> None:
        """Validate chunking configuration."""
        if self.config.min_chunk_size <= 0:
            raise InvalidChunkSizeError("min_chunk_size must be positive")
        
        if self.config.max_chunk_size <= self.config.min_chunk_size:
            raise InvalidChunkSizeError("max_chunk_size must be greater than min_chunk_size")
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk source code text.
        
        Args:
            text: Source code content
            metadata: Optional metadata (should include 'programming_language')
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        language = metadata.get('programming_language', 'unknown')
        
        # Detect language from metadata or content
        if language == 'unknown':
            language = self._detect_language(text, metadata)
        
        logger.debug(f"Chunking code with language: {language}")
        
        # For now, use a simple approach: chunk by code blocks in markdown
        # or by line count for raw code
        if self._is_markdown_formatted(text):
            chunks = self._chunk_markdown_code(text, metadata)
        else:
            chunks = self._chunk_raw_code(text, metadata, language)
        
        return chunks
    
    def _detect_language(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Detect programming language from text or metadata.
        
        Args:
            text: Source code content
            metadata: Document metadata
            
        Returns:
            Language name
        """
        # Try to get from file extension
        if 'file_extension' in metadata:
            ext = metadata['file_extension'].lower()
            ext_map = {
                '.js': 'javascript',
                '.jsx': 'javascript',
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.py': 'python',
                '.java': 'java',
                '.go': 'go',
                '.rs': 'rust',
                '.rb': 'ruby',
                '.php': 'php',
            }
            if ext in ext_map:
                return ext_map[ext]
        
        # Try to detect from content patterns
        if 'def ' in text and ':' in text:
            return 'python'
        elif 'function ' in text or 'const ' in text or '=>' in text:
            return 'javascript'
        elif 'public class ' in text or 'private class ' in text:
            return 'java'
        
        return 'unknown'
    
    def _is_markdown_formatted(self, text: str) -> bool:
        """
        Check if text is markdown-formatted code.
        
        Args:
            text: Text to check
            
        Returns:
            True if markdown formatted
        """
        # Check for markdown code blocks
        return bool(re.search(r'```[\w]*\n', text))
    
    def _chunk_markdown_code(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk markdown-formatted code.
        
        Args:
            text: Markdown text with code blocks
            metadata: Document metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Extract code blocks from markdown
        code_block_pattern = re.compile(r'```([\w]*)\n(.*?)```', re.DOTALL)
        matches = list(code_block_pattern.finditer(text))
        
        if not matches:
            # No code blocks found, treat as single chunk
            return [Chunk(
                content=text,
                chunk_type=ChunkType.CODE,
                metadata=metadata,
                start_index=0,
                end_index=len(text)
            )]

        # Extract header (everything before first code block)
        first_match = matches[0]
        header = text[:first_match.start()].strip()

        # Create chunks for each code block
        for i, match in enumerate(matches):
            language = match.group(1) or 'unknown'
            code_content = match.group(2).strip()

            # Include header with first chunk
            if i == 0 and header:
                chunk_content = f"{header}\n\n```{language}\n{code_content}\n```"
            else:
                chunk_content = f"```{language}\n{code_content}\n```"

            # Check if chunk is too large
            if len(chunk_content) > self.config.max_chunk_size:
                # Split large code blocks
                sub_chunks = self._split_large_code_block(
                    code_content, language, metadata, match.start()
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=ChunkType.CODE,
                    metadata=metadata,
                    start_index=match.start(),
                    end_index=match.end()
                ))

        return chunks

    def _chunk_raw_code(
        self,
        text: str,
        metadata: Dict[str, Any],
        language: str
    ) -> List[Chunk]:
        """
        Chunk raw source code (not markdown formatted).

        Args:
            text: Raw source code
            metadata: Document metadata
            language: Programming language

        Returns:
            List of chunks
        """
        # For raw code, try to split by functions/classes
        if language == 'python':
            return self._chunk_python_code(text, metadata)
        elif language in ['javascript', 'typescript']:
            return self._chunk_javascript_code(text, metadata)
        else:
            # Fallback: split by line count
            return self._chunk_by_lines(text, metadata)

    def _chunk_python_code(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk Python code by functions and classes.

        Args:
            text: Python source code
            metadata: Document metadata

        Returns:
            List of chunks
        """
        chunks = []
        lines = text.split('\n')

        # Simple pattern matching for Python functions and classes
        current_chunk_lines = []
        current_start = 0
        in_definition = False
        indent_level = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # Detect function or class definition
            if stripped.startswith('def ') or stripped.startswith('class '):
                # Save previous chunk if exists
                if current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines)
                    if chunk_content.strip():
                        chunks.append(Chunk(
                            content=chunk_content,
                            chunk_type=ChunkType.CODE,
                            metadata=metadata,
                            start_index=current_start,
                            end_index=i
                        ))

                # Start new chunk
                current_chunk_lines = [line]
                current_start = i
                in_definition = True
                indent_level = len(line) - len(stripped)

            elif in_definition:
                current_chunk_lines.append(line)

                # Check if we've exited the definition
                if stripped and not line.startswith(' ' * (indent_level + 1)):
                    if not stripped.startswith('def ') and not stripped.startswith('class '):
                        in_definition = False
            else:
                current_chunk_lines.append(line)

        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=ChunkType.CODE,
                    metadata=metadata,
                    start_index=current_start,
                    end_index=len(lines)
                ))

        return chunks if chunks else [Chunk(
            content=text,
            chunk_type=ChunkType.CODE,
            metadata=metadata,
            start_index=0,
            end_index=len(text)
        )]

    def _chunk_javascript_code(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk JavaScript/TypeScript code by functions and classes.

        Args:
            text: JavaScript/TypeScript source code
            metadata: Document metadata

        Returns:
            List of chunks
        """
        chunks = []
        lines = text.split('\n')

        # Simple pattern matching for JS functions and classes
        current_chunk_lines = []
        current_start = 0
        brace_count = 0
        in_definition = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect function or class definition
            if (stripped.startswith('function ') or
                stripped.startswith('class ') or
                stripped.startswith('const ') and '=>' in line or
                stripped.startswith('export function ') or
                stripped.startswith('export class ')):

                # Save previous chunk if exists
                if current_chunk_lines and brace_count == 0:
                    chunk_content = '\n'.join(current_chunk_lines)
                    if chunk_content.strip():
                        chunks.append(Chunk(
                            content=chunk_content,
                            chunk_type=ChunkType.CODE,
                            metadata=metadata,
                            start_index=current_start,
                            end_index=i
                        ))

                    # Start new chunk
                    current_chunk_lines = []
                    current_start = i

                in_definition = True

            current_chunk_lines.append(line)

            # Track braces to know when definition ends
            brace_count += line.count('{') - line.count('}')

            # If we've closed all braces, definition is complete
            if in_definition and brace_count == 0 and '{' in ''.join(current_chunk_lines):
                in_definition = False

        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=ChunkType.CODE,
                    metadata=metadata,
                    start_index=current_start,
                    end_index=len(lines)
                ))

        return chunks if chunks else [Chunk(
            content=text,
            chunk_type=ChunkType.CODE,
            metadata=metadata,
            start_index=0,
            end_index=len(text)
        )]

    def _chunk_by_lines(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Fallback: chunk code by line count.

        Args:
            text: Source code
            metadata: Document metadata

        Returns:
            List of chunks
        """
        lines = text.split('\n')
        chunks = []

        # Estimate lines per chunk based on average line length
        avg_line_length = len(text) / len(lines) if lines else 80
        lines_per_chunk = max(10, int(self.config.max_chunk_size / avg_line_length))

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)

            if chunk_content.strip():
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=ChunkType.CODE,
                    metadata=metadata,
                    start_index=i,
                    end_index=min(i + lines_per_chunk, len(lines))
                ))

        return chunks

    def _split_large_code_block(
        self,
        code: str,
        language: str,
        metadata: Dict[str, Any],
        start_index: int
    ) -> List[Chunk]:
        """
        Split a large code block into smaller chunks.

        Args:
            code: Code content
            language: Programming language
            metadata: Document metadata
            start_index: Starting position in original text

        Returns:
            List of chunks
        """
        lines = code.split('\n')
        chunks = []

        # Calculate lines per chunk
        avg_line_length = len(code) / len(lines) if lines else 80
        lines_per_chunk = max(10, int(self.config.max_chunk_size / avg_line_length))

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_content = f"```{language}\n" + '\n'.join(chunk_lines) + "\n```"

            chunks.append(Chunk(
                content=chunk_content,
                chunk_type=ChunkType.CODE,
                metadata=metadata,
                start_index=start_index + i,
                end_index=start_index + min(i + lines_per_chunk, len(lines))
            ))

        return chunks

