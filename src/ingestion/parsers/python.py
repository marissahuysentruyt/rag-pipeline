"""
Python code parser using the built-in ast module.

This parser extracts functions, classes, and other entities from Python source code.
"""

import ast
import logging
from typing import List, Optional

from .base import CodeParser, CodeEntity, EntityType, ParseError

logger = logging.getLogger(__name__)


class PythonParser(CodeParser):
    """Parser for Python source code."""
    
    def __init__(self):
        """Initialize Python parser."""
        super().__init__("python")
    
    def parse(self, source_code: str, file_path: Optional[str] = None) -> List[CodeEntity]:
        """
        Parse Python source code and extract entities.
        
        Args:
            source_code: Python source code
            file_path: Optional file path for context
            
        Returns:
            List of CodeEntity objects
            
        Raises:
            ParseError: If parsing fails
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ParseError(f"Failed to parse Python code: {e}") from e
        
        entities = []
        lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                entity = self._extract_function(node, lines)
                entities.append(entity)
            elif isinstance(node, ast.ClassDef):
                entity = self._extract_class(node, lines)
                entities.append(entity)
        
        return entities
    
    def _extract_function(self, node: ast.FunctionDef, lines: List[str]) -> CodeEntity:
        """
        Extract function entity from AST node.
        
        Args:
            node: FunctionDef AST node
            lines: Source code lines
            
        Returns:
            CodeEntity for the function
        """
        # Extract function content
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        content = '\n'.join(lines[start_line:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]
        
        # Extract decorators
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        # Build signature
        params_str = ', '.join(parameters)
        return_annotation = ''
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"
        signature = f"def {node.name}({params_str}){return_annotation}"
        
        return CodeEntity(
            name=node.name,
            entity_type=EntityType.FUNCTION,
            content=content,
            docstring=docstring,
            start_line=start_line + 1,
            end_line=end_line,
            signature=signature,
            parameters=parameters,
            return_type=ast.unparse(node.returns) if node.returns else None,
            decorators=decorators if decorators else None
        )
    
    def _extract_class(self, node: ast.ClassDef, lines: List[str]) -> CodeEntity:
        """
        Extract class entity from AST node.
        
        Args:
            node: ClassDef AST node
            lines: Source code lines
            
        Returns:
            CodeEntity for the class
        """
        # Extract class content
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        content = '\n'.join(lines[start_line:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract decorators
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        # Build signature
        bases = [ast.unparse(base) for base in node.bases]
        bases_str = f"({', '.join(bases)})" if bases else ""
        signature = f"class {node.name}{bases_str}"
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        
        return CodeEntity(
            name=node.name,
            entity_type=EntityType.CLASS,
            content=content,
            docstring=docstring,
            start_line=start_line + 1,
            end_line=end_line,
            signature=signature,
            decorators=decorators if decorators else None,
            metadata={"methods": methods} if methods else None
        )

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """
        Get decorator name from AST node.

        Args:
            decorator: Decorator AST node

        Returns:
            Decorator name as string
        """
        try:
            return ast.unparse(decorator)
        except:
            return str(decorator)

    def extract_imports(self, source_code: str) -> List[str]:
        """
        Extract import statements from Python code.

        Args:
            source_code: Python source code

        Returns:
            List of import statements
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = ', '.join([alias.name for alias in node.names])
                imports.append(f"from {module} import {names}")

        return imports

    def extract_docstring(self, entity_code: str) -> Optional[str]:
        """
        Extract docstring from Python entity code.

        Args:
            entity_code: Python code of a single entity

        Returns:
            Docstring text or None
        """
        try:
            tree = ast.parse(entity_code)
            return ast.get_docstring(tree)
        except:
            return None


