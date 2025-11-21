"""
JavaScript/TypeScript code parser using regex patterns.

This is a simple parser that uses regex to extract functions and classes.
For production use, consider using a proper parser like esprima or tree-sitter.
"""

import re
import logging
from typing import List, Optional

from .base import CodeParser, CodeEntity, EntityType, ParseError

logger = logging.getLogger(__name__)


class JavaScriptParser(CodeParser):
    """Simple parser for JavaScript/TypeScript source code."""
    
    def __init__(self, language: str = "javascript"):
        """
        Initialize JavaScript parser.
        
        Args:
            language: "javascript" or "typescript"
        """
        super().__init__(language)
    
    def parse(self, source_code: str, file_path: Optional[str] = None) -> List[CodeEntity]:
        """
        Parse JavaScript/TypeScript source code and extract entities.
        
        Args:
            source_code: JavaScript/TypeScript source code
            file_path: Optional file path for context
            
        Returns:
            List of CodeEntity objects
        """
        entities = []
        lines = source_code.split('\n')
        
        # Extract functions
        entities.extend(self._extract_functions(source_code, lines))
        
        # Extract classes
        entities.extend(self._extract_classes(source_code, lines))
        
        # Extract arrow functions assigned to const/let
        entities.extend(self._extract_arrow_functions(source_code, lines))
        
        return entities
    
    def _extract_functions(self, source_code: str, lines: List[str]) -> List[CodeEntity]:
        """Extract function declarations."""
        entities = []
        
        # Pattern for function declarations
        # Matches: function name(...) { ... }
        # Also matches: export function name(...) { ... }
        pattern = r'(export\s+)?(async\s+)?function\s+(\w+)\s*\((.*?)\)'
        
        for match in re.finditer(pattern, source_code):
            func_name = match.group(3)
            params = match.group(4)
            start_pos = match.start()
            
            # Find the function body
            start_line = source_code[:start_pos].count('\n')
            end_line = self._find_closing_brace(source_code, match.end(), lines)
            
            if end_line:
                content = '\n'.join(lines[start_line:end_line + 1])
                
                # Extract JSDoc comment if present
                docstring = self._extract_jsdoc(lines, start_line)
                
                # Build signature
                is_async = bool(match.group(2))
                is_export = bool(match.group(1))
                signature = f"{'export ' if is_export else ''}{'async ' if is_async else ''}function {func_name}({params})"
                
                entities.append(CodeEntity(
                    name=func_name,
                    entity_type=EntityType.FUNCTION,
                    content=content,
                    docstring=docstring,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    signature=signature,
                    parameters=self._parse_parameters(params)
                ))
        
        return entities
    
    def _extract_classes(self, source_code: str, lines: List[str]) -> List[CodeEntity]:
        """Extract class declarations."""
        entities = []
        
        # Pattern for class declarations
        pattern = r'(export\s+)?(default\s+)?class\s+(\w+)(\s+extends\s+\w+)?'
        
        for match in re.finditer(pattern, source_code):
            class_name = match.group(3)
            start_pos = match.start()
            
            # Find the class body
            start_line = source_code[:start_pos].count('\n')
            end_line = self._find_closing_brace(source_code, match.end(), lines)
            
            if end_line:
                content = '\n'.join(lines[start_line:end_line + 1])
                
                # Extract JSDoc comment if present
                docstring = self._extract_jsdoc(lines, start_line)
                
                # Build signature
                is_export = bool(match.group(1))
                is_default = bool(match.group(2))
                extends = match.group(4) or ''
                signature = f"{'export ' if is_export else ''}{'default ' if is_default else ''}class {class_name}{extends}"
                
                entities.append(CodeEntity(
                    name=class_name,
                    entity_type=EntityType.CLASS,
                    content=content,
                    docstring=docstring,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    signature=signature
                ))
        
        return entities
    
    def _extract_arrow_functions(self, source_code: str, lines: List[str]) -> List[CodeEntity]:
        """Extract arrow functions assigned to variables."""
        entities = []
        
        # Pattern for arrow functions
        # Matches: const name = (...) => { ... }
        pattern = r'(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(\(.*?\)|[\w]+)\s*=>'

        for match in re.finditer(pattern, source_code):
            func_name = match.group(3)
            params = match.group(4).strip('()')
            start_pos = match.start()

            start_line = source_code[:start_pos].count('\n')

            # Check if it's a block arrow function or expression
            arrow_pos = match.end()
            if arrow_pos < len(source_code) and source_code[arrow_pos:].lstrip().startswith('{'):
                end_line = self._find_closing_brace(source_code, arrow_pos, lines)
            else:
                # Expression arrow function - find end of statement
                end_line = self._find_statement_end(source_code, arrow_pos, lines)

            if end_line:
                content = '\n'.join(lines[start_line:end_line + 1])

                # Extract JSDoc comment if present
                docstring = self._extract_jsdoc(lines, start_line)

                is_export = bool(match.group(1))
                signature = f"{'export ' if is_export else ''}const {func_name} = ({params}) =>"

                entities.append(CodeEntity(
                    name=func_name,
                    entity_type=EntityType.FUNCTION,
                    content=content,
                    docstring=docstring,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    signature=signature,
                    parameters=self._parse_parameters(params)
                ))

        return entities

    def _find_closing_brace(self, source_code: str, start_pos: int, lines: List[str]) -> Optional[int]:
        """Find the line number of the closing brace."""
        brace_count = 0
        in_string = False
        string_char = None

        for i in range(start_pos, len(source_code)):
            char = source_code[i]

            # Handle strings
            if char in ['"', "'", '`'] and (i == 0 or source_code[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return source_code[:i].count('\n')

        return None

    def _find_statement_end(self, source_code: str, start_pos: int, lines: List[str]) -> Optional[int]:
        """Find the end of a statement (semicolon or newline)."""
        for i in range(start_pos, len(source_code)):
            if source_code[i] in [';', '\n']:
                return source_code[:i].count('\n')
        return len(lines) - 1

    def _extract_jsdoc(self, lines: List[str], func_line: int) -> Optional[str]:
        """Extract JSDoc comment before a function/class."""
        if func_line == 0:
            return None

        # Look backwards for JSDoc comment
        jsdoc_lines = []
        for i in range(func_line - 1, -1, -1):
            line = lines[i].strip()
            if line.endswith('*/'):
                jsdoc_lines.insert(0, line)
                # Continue looking for start
                for j in range(i - 1, -1, -1):
                    jsdoc_lines.insert(0, lines[j].strip())
                    if lines[j].strip().startswith('/**'):
                        return '\n'.join(jsdoc_lines)
                break
            elif not line or line.startswith('//'):
                continue
            else:
                break

        return None

    def _parse_parameters(self, params_str: str) -> Optional[List[str]]:
        """Parse parameter string into list of parameter names."""
        if not params_str or not params_str.strip():
            return None

        # Simple split by comma (doesn't handle complex cases)
        params = []
        for param in params_str.split(','):
            param = param.strip()
            # Remove default values and type annotations
            param = re.sub(r'[:=].*', '', param).strip()
            if param:
                params.append(param)

        return params if params else None

    def extract_imports(self, source_code: str) -> List[str]:
        """Extract import statements."""
        imports = []

        # ES6 imports
        import_pattern = r'import\s+.*?from\s+["\'].*?["\']'
        for match in re.finditer(import_pattern, source_code):
            imports.append(match.group(0))

        # require statements
        require_pattern = r'(const|let|var)\s+.*?=\s*require\(["\'].*?["\']\)'
        for match in re.finditer(require_pattern, source_code):
            imports.append(match.group(0))

        return imports

    def extract_docstring(self, entity_code: str) -> Optional[str]:
        """Extract JSDoc comment from entity code."""
        lines = entity_code.split('\n')
        return self._extract_jsdoc(lines, len(lines) - 1)


