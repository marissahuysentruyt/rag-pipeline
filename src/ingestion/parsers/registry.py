"""
Code parser registry for mapping languages to parser implementations.

This module provides a registry system for managing code parsers across
multiple programming languages with lazy loading and fallback support.
"""

import logging
from typing import Dict, Optional, Type
from pathlib import Path

from .base import CodeParser
from .python import PythonParser
from .javascript import JavaScriptParser

logger = logging.getLogger(__name__)


class CodeParserRegistry:
    """
    Registry for managing code parsers across multiple languages.

    Provides a centralized system for:
    - Mapping file extensions and language names to parser implementations
    - Lazy-loading parsers for performance
    - Detecting language from file extensions
    - Fallback handling for unsupported languages

    Example:
        >>> registry = CodeParserRegistry()
        >>> parser = registry.get_parser("python")
        >>> entities = parser.parse(source_code)
    """

    # Map of file extensions to language names
    EXTENSION_MAP = {
        '.py': 'python',
        '.pyw': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.cs': 'csharp',
    }

    # Map of language names to language families (for parser selection)
    LANGUAGE_FAMILIES = {
        'javascript': 'javascript',
        'typescript': 'javascript',  # TypeScript uses JavaScript parser
        'jsx': 'javascript',
        'tsx': 'javascript',
    }

    def __init__(self):
        """Initialize the parser registry."""
        self._parsers: Dict[str, CodeParser] = {}
        self._parser_classes: Dict[str, Type[CodeParser]] = {}

        # Register built-in parsers
        self._register_builtin_parsers()

    def _register_builtin_parsers(self):
        """Register built-in parser implementations."""
        self.register_parser('python', PythonParser)
        self.register_parser('javascript', JavaScriptParser)
        self.register_parser('typescript', JavaScriptParser)  # TypeScript uses JS parser

    def register_parser(self, language: str, parser_class: Type[CodeParser]):
        """
        Register a parser class for a language.

        Args:
            language: Language name (e.g., "python", "javascript")
            parser_class: Parser class (not instance) to use for this language
        """
        language = language.lower()
        self._parser_classes[language] = parser_class
        logger.debug(f"Registered parser for language: {language}")

    def get_parser(self, language: str) -> Optional[CodeParser]:
        """
        Get parser for a language, creating instance if needed.

        Args:
            language: Language name (e.g., "python", "javascript")

        Returns:
            Parser instance or None if language not supported
        """
        language = language.lower()

        # Map to language family if needed
        language = self.LANGUAGE_FAMILIES.get(language, language)

        # Return cached parser if available
        if language in self._parsers:
            return self._parsers[language]

        # Create new parser instance if class is registered
        if language in self._parser_classes:
            parser_class = self._parser_classes[language]
            try:
                # Instantiate parser (handle both with/without language parameter)
                if language == 'python':
                    parser = parser_class()
                else:
                    parser = parser_class(language)

                self._parsers[language] = parser
                logger.debug(f"Created parser instance for language: {language}")
                return parser
            except Exception as e:
                logger.error(f"Failed to instantiate parser for {language}: {e}")
                return None

        logger.warning(f"No parser registered for language: {language}")
        return None

    def get_parser_for_file(self, file_path: str) -> Optional[CodeParser]:
        """
        Get parser based on file extension.

        Args:
            file_path: Path to code file

        Returns:
            Parser instance or None if extension not recognized
        """
        extension = Path(file_path).suffix.lower()

        if extension not in self.EXTENSION_MAP:
            logger.debug(f"No language mapping for extension: {extension}")
            return None

        language = self.EXTENSION_MAP[extension]
        return self.get_parser(language)

    def supports_language(self, language: str) -> bool:
        """
        Check if a language is supported.

        Args:
            language: Language name

        Returns:
            True if parser is available for this language
        """
        language = language.lower()
        language = self.LANGUAGE_FAMILIES.get(language, language)
        return language in self._parser_classes

    def supports_file(self, file_path: str) -> bool:
        """
        Check if a file extension is supported.

        Args:
            file_path: Path to code file

        Returns:
            True if parser is available for this file type
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.EXTENSION_MAP

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported language names.

        Returns:
            List of language names with registered parsers
        """
        return list(self._parser_classes.keys())

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of file extensions (e.g., ['.py', '.js'])
        """
        return list(self.EXTENSION_MAP.keys())

    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect language from file extension.

        Args:
            file_path: Path to code file

        Returns:
            Language name or None if not recognized
        """
        extension = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(extension)
