"""
Code entity formatter for converting parsed entities to documents.

This module converts CodeEntity objects from parsers into Document objects
with rich metadata suitable for indexing and retrieval.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from ..parsers.base import CodeEntity, EntityType
from ..adapters.base import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class CodeEntityFormatter:
    """
    Formats CodeEntity objects into Document objects for indexing.

    Converts parsed code entities (functions, classes, methods) into
    documents with rich metadata including signatures, parameters,
    return types, and hierarchical information.

    Example:
        >>> formatter = CodeEntityFormatter()
        >>> entity = CodeEntity(name="my_function", entity_type=EntityType.FUNCTION, ...)
        >>> document = formatter.format_entity(entity, "src/utils.py", "python")
        >>> document.metadata.additional_metadata["entity_type"]
        'function'
    """

    def __init__(self):
        """Initialize the code entity formatter."""
        pass

    def format_entity(
        self,
        entity: CodeEntity,
        file_path: str,
        language: str,
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Convert a CodeEntity to a Document with rich metadata.

        Args:
            entity: Parsed code entity (function, class, method, etc.)
            file_path: Path to the source file
            language: Programming language name
            file_metadata: Optional additional file-level metadata

        Returns:
            Document object ready for indexing
        """
        # Build document ID (unique identifier)
        source_id = self._build_source_id(file_path, entity)

        # Build title for display
        title = self._build_title(entity, file_path)

        # Build content (formatted entity code)
        content = self._build_content(entity, language)

        # Build comprehensive metadata
        additional_metadata = self._build_metadata(entity, file_path, language)

        # Merge with file-level metadata if provided
        if file_metadata:
            additional_metadata.update(file_metadata)

        # Create DocumentMetadata
        metadata = DocumentMetadata(
            source_id=source_id,
            title=title,
            url=None,  # Code entities don't have URLs
            source_type="codebase",
            last_modified=datetime.now(),
            language=language,
            additional_metadata=additional_metadata
        )

        # Create and return Document
        return Document(
            content=content,
            metadata=metadata,
            format="markdown"  # Format code as markdown with syntax highlighting
        )

    def format_entities(
        self,
        entities: list[CodeEntity],
        file_path: str,
        language: str,
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> list[Document]:
        """
        Convert multiple CodeEntity objects to Documents.

        Args:
            entities: List of parsed code entities
            file_path: Path to the source file
            language: Programming language name
            file_metadata: Optional additional file-level metadata

        Returns:
            List of Document objects ready for indexing
        """
        documents = []
        for entity in entities:
            try:
                doc = self.format_entity(entity, file_path, language, file_metadata)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to format entity {entity.name} from {file_path}: {e}")
                continue

        return documents

    def _build_source_id(self, file_path: str, entity: CodeEntity) -> str:
        """
        Build unique source ID for entity.

        Format: file_path::entity_name or file_path::ParentClass::method_name

        Args:
            file_path: Path to source file
            entity: Code entity

        Returns:
            Unique identifier string
        """
        if entity.parent:
            return f"{file_path}::{entity.parent}::{entity.name}"
        else:
            return f"{file_path}::{entity.name}"

    def _build_title(self, entity: CodeEntity, file_path: str) -> str:
        """
        Build display title for entity.

        Args:
            entity: Code entity
            file_path: Path to source file

        Returns:
            Human-readable title
        """
        entity_type_name = entity.entity_type.value.capitalize()

        if entity.parent:
            return f"{entity_type_name}: {entity.parent}.{entity.name}"
        else:
            return f"{entity_type_name}: {entity.name}"

    def _build_content(self, entity: CodeEntity, language: str) -> str:
        """
        Build markdown-formatted content for entity.

        Args:
            entity: Code entity
            language: Programming language name

        Returns:
            Markdown-formatted content with code block
        """
        lines = []

        # Add entity header
        lines.append(f"# {entity.name}")
        lines.append("")

        # Add signature if available
        if entity.signature:
            lines.append(f"**Signature:** `{entity.signature}`")
            lines.append("")

        # Add docstring if available
        if entity.docstring:
            lines.append("## Documentation")
            lines.append("")
            lines.append(entity.docstring)
            lines.append("")

        # Add parameters if available
        if entity.parameters:
            lines.append("**Parameters:**")
            for param in entity.parameters:
                lines.append(f"- `{param}`")
            lines.append("")

        # Add return type if available
        if entity.return_type:
            lines.append(f"**Returns:** `{entity.return_type}`")
            lines.append("")

        # Add decorators if available
        if entity.decorators:
            lines.append("**Decorators:**")
            for decorator in entity.decorators:
                lines.append(f"- `@{decorator}`")
            lines.append("")

        # Add source code
        lines.append("## Source Code")
        lines.append("")
        lines.append(f"```{language}")
        lines.append(entity.content)
        lines.append("```")

        return "\n".join(lines)

    def _build_metadata(
        self,
        entity: CodeEntity,
        file_path: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata dictionary for entity.

        Args:
            entity: Code entity
            file_path: Path to source file
            language: Programming language name

        Returns:
            Dictionary of metadata fields
        """
        metadata = {
            "entity_type": entity.entity_type.value,
            "entity_name": entity.name,
            "file_path": file_path,
            "programming_language": language,
        }

        # Add optional fields if present
        if entity.signature:
            metadata["signature"] = entity.signature

        if entity.parameters:
            metadata["parameters"] = entity.parameters

        if entity.return_type:
            metadata["return_type"] = entity.return_type

        if entity.decorators:
            metadata["decorators"] = entity.decorators

        if entity.parent:
            metadata["parent_entity"] = entity.parent

        if entity.docstring:
            metadata["has_docstring"] = True

        if entity.start_line is not None:
            metadata["start_line"] = entity.start_line

        if entity.end_line is not None:
            metadata["end_line"] = entity.end_line

        if entity.metadata:
            # Merge any additional entity-specific metadata
            metadata.update(entity.metadata)

        return metadata
