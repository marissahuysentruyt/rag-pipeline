"""
Codebase adapter for ingesting source code files.

This adapter walks through a codebase directory, extracts code entities
(functions, classes, components), and creates documents for indexing.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator

from .base import (
    IngestionAdapter,
    Document,
    DocumentMetadata,
    IngestionError,
    DocumentNotFoundError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class CodebaseAdapter(IngestionAdapter):
    """
    Adapter for ingesting source code from a local codebase.
    
    This adapter:
    1. Walks through a directory tree
    2. Filters files by extension
    3. Extracts code entities (functions, classes, etc.)
    4. Creates documents with rich metadata
    
    Configuration:
        repo_path: Path to the codebase root
        file_patterns: List of glob patterns to include (e.g., ["**/*.js", "**/*.py"])
        exclude_patterns: List of glob patterns to exclude (e.g., ["**/node_modules/**"])
        languages: List of languages to parse (e.g., ["javascript", "python"])
        extract_entities: List of entity types to extract (e.g., ["functions", "classes"])
    """
    
    # Supported file extensions by language
    LANGUAGE_EXTENSIONS = {
        "javascript": [".js", ".jsx", ".mjs"],
        "typescript": [".ts", ".tsx"],
        "python": [".py"],
        "java": [".java"],
        "go": [".go"],
        "rust": [".rs"],
        "ruby": [".rb"],
        "php": [".php"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".hpp", ".cc", ".hh"],
        "csharp": [".cs"],
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the codebase adapter.
        
        Args:
            config: Configuration dictionary with repo_path, file_patterns, etc.
        """
        self.repo_path: Optional[Path] = None
        self.file_patterns: List[str] = []
        self.exclude_patterns: List[str] = []
        self.languages: List[str] = []
        self.extract_entities: List[str] = []
        self._file_cache: Dict[str, Document] = {}
        
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if "repo_path" not in self.config:
            raise ConfigurationError("repo_path is required")
        
        repo_path = Path(self.config["repo_path"])
        if not repo_path.exists():
            raise ConfigurationError(f"Repository path does not exist: {repo_path}")
        
        if not repo_path.is_dir():
            raise ConfigurationError(f"Repository path is not a directory: {repo_path}")
        
        self.repo_path = repo_path
        self.file_patterns = self.config.get("file_patterns", ["**/*.js", "**/*.py"])
        self.exclude_patterns = self.config.get("exclude_patterns", [
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/*.test.*",
            "**/*.spec.*"
        ])
        self.languages = self.config.get("languages", ["javascript", "python"])
        self.extract_entities = self.config.get("extract_entities", ["functions", "classes"])
        
        logger.info(f"Codebase adapter configured for: {self.repo_path}")
        logger.info(f"Languages: {', '.join(self.languages)}")
        logger.info(f"File patterns: {', '.join(self.file_patterns)}")
    
    def connect(self) -> None:
        """Establish connection (verify directory access)."""
        if not self.repo_path or not self.repo_path.exists():
            raise ConnectionError(f"Cannot access repository: {self.repo_path}")
        
        logger.info(f"Connected to codebase at: {self.repo_path}")
    
    def disconnect(self) -> None:
        """Cleanup resources."""
        self._file_cache.clear()
        logger.info("Disconnected from codebase")
    
    def _should_include_file(self, file_path: Path) -> bool:
        """
        Check if a file should be included based on patterns.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be included
        """
        relative_path = file_path.relative_to(self.repo_path)
        path_str = str(relative_path)
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return False

        # Check if extension matches configured languages
        extension = file_path.suffix.lower()
        for language in self.languages:
            if extension in self.LANGUAGE_EXTENSIONS.get(language, []):
                return True

        return False

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language name or None
        """
        extension = file_path.suffix.lower()
        for language, extensions in self.LANGUAGE_EXTENSIONS.items():
            if extension in extensions:
                return language
        return None

    def _extract_file_content(self, file_path: Path) -> str:
        """
        Read file content.

        Args:
            file_path: Path to file

        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                return ""
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""

    def _create_document_from_file(self, file_path: Path) -> Document:
        """
        Create a Document from a code file.

        Args:
            file_path: Path to code file

        Returns:
            Document object
        """
        content = self._extract_file_content(file_path)
        if not content:
            raise DocumentNotFoundError(f"Could not read content from {file_path}")

        relative_path = file_path.relative_to(self.repo_path)
        language = self._detect_language(file_path)

        # Get file stats
        stats = file_path.stat()
        last_modified = datetime.fromtimestamp(stats.st_mtime)

        # Create metadata
        metadata = DocumentMetadata(
            source_id=str(relative_path),
            source_type="codebase",
            title=file_path.name,
            url=None,  # Could add GitHub URL if available
            last_modified=last_modified,
            language=language or "unknown",
            additional_metadata={
                "file_path": str(relative_path),
                "absolute_path": str(file_path),
                "file_extension": file_path.suffix,
                "file_size": stats.st_size,
                "programming_language": language,
                "repo_path": str(self.repo_path),
            }
        )

        # Format content as markdown with syntax highlighting
        formatted_content = self._format_as_markdown(content, language, relative_path)

        return Document(
            content=formatted_content,
            metadata=metadata,
            format="markdown"
        )

    def _format_as_markdown(self, content: str, language: Optional[str], file_path: Path) -> str:
        """
        Format code content as markdown with syntax highlighting.

        Args:
            content: Raw code content
            language: Programming language
            file_path: Relative file path

        Returns:
            Markdown-formatted content
        """
        lang_tag = language or ""

        # Create markdown with metadata header and code block
        markdown = f"""# {file_path.name}

**File:** `{file_path}`
**Language:** {language or "Unknown"}

```{lang_tag}
{content}
```
"""
        return markdown

    def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List all code files in the repository.

        Args:
            filters: Optional filters (e.g., {"language": "python"})
            limit: Maximum number of files to return

        Returns:
            List of file paths (relative to repo root)
        """
        if not self.repo_path:
            return []

        files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and self._should_include_file(file_path):
                relative_path = str(file_path.relative_to(self.repo_path))

                # Apply filters if provided
                if filters:
                    language = self._detect_language(file_path)
                    if "language" in filters and language != filters["language"]:
                        continue

                files.append(relative_path)

                if limit and len(files) >= limit:
                    break

        logger.info(f"Found {len(files)} code files in {self.repo_path}")
        return files

    def fetch_document(self, document_id: str) -> Document:
        """
        Fetch a single code file by its path.

        Args:
            document_id: Relative file path from repo root

        Returns:
            Document object

        Raises:
            DocumentNotFoundError: If file doesn't exist
        """
        if not self.repo_path:
            raise DocumentNotFoundError("Repository not connected")

        file_path = self.repo_path / document_id

        if not file_path.exists():
            raise DocumentNotFoundError(f"File not found: {document_id}")

        if not file_path.is_file():
            raise DocumentNotFoundError(f"Not a file: {document_id}")

        return self._create_document_from_file(file_path)

    def fetch_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 10
    ) -> Iterator[Document]:
        """
        Fetch all code files from the repository.

        Args:
            filters: Optional filters (e.g., {"language": "python"})
            batch_size: Number of documents to process at once

        Yields:
            Document objects
        """
        document_ids = self.list_documents(filters=filters)

        logger.info(f"Fetching {len(document_ids)} documents from codebase")

        for i, doc_id in enumerate(document_ids):
            try:
                document = self.fetch_document(doc_id)
                yield document

                if (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1}/{len(document_ids)} files")

            except Exception as e:
                logger.error(f"Error processing {doc_id}: {e}")
                continue

        logger.info(f"Completed fetching {len(document_ids)} documents")

    def get_updates_since(
        self,
        timestamp: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Iterator[Document]:
        """
        Fetch only files modified since a given timestamp.

        Args:
            timestamp: Only fetch files modified after this time
            filters: Optional additional filters

        Yields:
            Updated Document objects
        """
        document_ids = self.list_documents(filters=filters)

        updated_count = 0
        for doc_id in document_ids:
            try:
                file_path = self.repo_path / doc_id
                stats = file_path.stat()
                last_modified = datetime.fromtimestamp(stats.st_mtime)

                if last_modified > timestamp:
                    document = self.fetch_document(doc_id)
                    yield document
                    updated_count += 1

            except Exception as e:
                logger.error(f"Error checking {doc_id}: {e}")
                continue

        logger.info(f"Found {updated_count} files updated since {timestamp}")

    def supports_incremental_updates(self) -> bool:
        """Check if adapter supports incremental updates."""
        return True

    def parse_with_entities(
        self,
        file_path: str,
        parser_registry: Optional[Any] = None,
        formatter: Optional[Any] = None
    ) -> List[Document]:
        """
        Parse a code file into individual entity documents.

        This method extracts code entities (functions, classes, methods) from
        a source file and converts each entity into a separate Document with
        rich metadata.

        Args:
            file_path: Path to code file (relative to repo_path)
            parser_registry: CodeParserRegistry instance (optional, creates default if None)
            formatter: CodeEntityFormatter instance (optional, creates default if None)

        Returns:
            List of Document objects, one per entity

        Raises:
            DocumentNotFoundError: If file doesn't exist
            IngestionError: If parsing fails
        """
        from ..parsers.registry import CodeParserRegistry
        from ..formatters.code_entity_formatter import CodeEntityFormatter

        # Create default registry and formatter if not provided
        if parser_registry is None:
            parser_registry = CodeParserRegistry()

        if formatter is None:
            formatter = CodeEntityFormatter()

        # Resolve file path
        if self.repo_path:
            full_path = self.repo_path / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            raise DocumentNotFoundError(f"File not found: {full_path}")

        # Detect language
        language = self._detect_language(full_path)
        if not language:
            logger.warning(f"Could not detect language for: {file_path}")
            return []

        # Get parser for this language
        parser = parser_registry.get_parser(language)
        if not parser:
            logger.warning(f"No parser available for language: {language}")
            return []

        try:
            # Read source code
            with open(full_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Parse into entities
            entities = parser.parse(source_code, str(full_path))

            if not entities:
                logger.debug(f"No entities found in: {file_path}")
                return []

            # Get file metadata for all entities
            file_metadata = {
                "file_size": full_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(full_path.stat().st_mtime)
            }

            # Convert entities to documents
            documents = formatter.format_entities(
                entities,
                file_path,
                language,
                file_metadata
            )

            logger.info(f"Extracted {len(documents)} entities from {file_path}")
            return documents

        except Exception as e:
            raise IngestionError(f"Failed to parse {file_path}: {e}") from e

    def fetch_all_entities(
        self,
        parser_registry: Optional[Any] = None,
        formatter: Optional[Any] = None,
        batch_size: int = 10
    ) -> Iterator[Document]:
        """
        Fetch all code files and parse into entity-level documents.

        This is an alternative to fetch_all() that returns entity-level
        documents instead of file-level documents.

        Args:
            parser_registry: CodeParserRegistry instance (optional)
            formatter: CodeEntityFormatter instance (optional)
            batch_size: Number of files to process before yielding (for logging)

        Yields:
            Document objects for each code entity
        """
        document_ids = self.list_documents()
        total_files = len(document_ids)
        total_entities = 0
        processed_files = 0

        logger.info(f"Processing {total_files} files for entity extraction...")

        for doc_id in document_ids:
            try:
                # Parse file into entity documents
                entity_docs = self.parse_with_entities(doc_id, parser_registry, formatter)

                # Yield each entity document
                for doc in entity_docs:
                    yield doc
                    total_entities += 1

                processed_files += 1

                # Log progress
                if processed_files % batch_size == 0:
                    logger.info(
                        f"Processed {processed_files}/{total_files} files, "
                        f"extracted {total_entities} entities"
                    )

            except Exception as e:
                logger.error(f"Error processing {doc_id}: {e}")
                continue

        logger.info(
            f"Completed: {processed_files} files processed, "
            f"{total_entities} entities extracted"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the codebase.

        Returns:
            Dictionary with stats
        """
        stats = super().get_stats()

        if self.repo_path:
            document_ids = self.list_documents()

            # Count files by language
            language_counts = {}
            total_size = 0

            for doc_id in document_ids:
                file_path = self.repo_path / doc_id
                language = self._detect_language(file_path)

                if language:
                    language_counts[language] = language_counts.get(language, 0) + 1

                try:
                    total_size += file_path.stat().st_size
                except:
                    pass

            stats.update({
                "repo_path": str(self.repo_path),
                "total_files": len(document_ids),
                "languages": language_counts,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            })

        return stats

