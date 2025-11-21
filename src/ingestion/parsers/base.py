"""
Base interface for code parsers.

This module defines the abstract base class for code parsers that extract
entities (functions, classes, etc.) from source code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class EntityType(Enum):
    """Types of code entities."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    TYPE = "type"
    COMPONENT = "component"  # For React/Vue components
    MODULE = "module"
    IMPORT = "import"


@dataclass
class CodeEntity:
    """
    Represents a code entity (function, class, etc.).
    
    Attributes:
        name: Entity name
        entity_type: Type of entity
        content: Full source code of the entity
        docstring: Documentation string if available
        start_line: Line number where entity starts
        end_line: Line number where entity ends
        signature: Function/method signature
        parameters: List of parameter names
        return_type: Return type if available
        decorators: List of decorators/annotations
        parent: Parent entity (e.g., class for a method)
        metadata: Additional entity-specific metadata
    """
    name: str
    entity_type: EntityType
    content: str
    docstring: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    signature: Optional[str] = None
    parameters: Optional[List[str]] = None
    return_type: Optional[str] = None
    decorators: Optional[List[str]] = None
    parent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CodeParser(ABC):
    """
    Abstract base class for code parsers.
    
    Each parser is responsible for:
    1. Parsing source code in a specific language
    2. Extracting code entities (functions, classes, etc.)
    3. Extracting docstrings and comments
    4. Providing entity metadata
    """
    
    def __init__(self, language: str):
        """
        Initialize the parser.
        
        Args:
            language: Programming language name
        """
        self.language = language
    
    @abstractmethod
    def parse(self, source_code: str, file_path: Optional[str] = None) -> List[CodeEntity]:
        """
        Parse source code and extract entities.
        
        Args:
            source_code: Source code to parse
            file_path: Optional file path for context
            
        Returns:
            List of CodeEntity objects
            
        Raises:
            ParseError: If parsing fails
        """
        pass
    
    @abstractmethod
    def extract_imports(self, source_code: str) -> List[str]:
        """
        Extract import statements from source code.
        
        Args:
            source_code: Source code to parse
            
        Returns:
            List of import statements
        """
        pass
    
    @abstractmethod
    def extract_docstring(self, entity_code: str) -> Optional[str]:
        """
        Extract docstring from entity code.
        
        Args:
            entity_code: Code of a single entity
            
        Returns:
            Docstring text or None
        """
        pass
    
    def get_language(self) -> str:
        """Get the programming language this parser handles."""
        return self.language
    
    def supports_language(self, language: str) -> bool:
        """
        Check if parser supports a given language.
        
        Args:
            language: Language name to check
            
        Returns:
            True if supported
        """
        return language.lower() == self.language.lower()


class ParseError(Exception):
    """Raised when code parsing fails."""
    pass

