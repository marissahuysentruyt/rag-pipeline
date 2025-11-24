"""
Integration tests for code entity extraction pipeline.

Tests the end-to-end flow:
1. CodebaseAdapter reads files
2. CodeParserRegistry provides parsers
3. Parsers extract entities
4. CodeEntityFormatter converts to Documents
5. Documents have correct metadata
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.ingestion.adapters import CodebaseAdapter
from src.ingestion.parsers import CodeParserRegistry, EntityType
from src.ingestion.formatters import CodeEntityFormatter


class TestCodeEntityIntegration:
    """Test end-to-end code entity extraction."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository with test files."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)

        # Create Python test file
        python_file = repo_path / "utils.py"
        python_file.write_text('''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

class Calculator:
    """A simple calculator class."""

    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

    def divide(self, x, y):
        """Divide two numbers."""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
''')

        # Create JavaScript test file
        js_file = repo_path / "helpers.js"
        js_file.write_text('''
function greet(name) {
    return `Hello, ${name}!`;
}

const add = (a, b) => {
    return a + b;
};

class Counter {
    constructor() {
        this.count = 0;
    }

    increment() {
        this.count++;
    }
}
''')

        yield repo_path

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_parse_python_file_with_entities(self, temp_repo):
        """Test parsing a Python file into entity documents."""
        # Setup adapter
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"],
            "file_patterns": ["**/*.py"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        # Parse file
        entity_docs = adapter.parse_with_entities("utils.py")

        # Should have 3 entities: 1 function + 1 class + 2 methods
        # Note: ast.walk() finds all nodes, so we get function + class + methods
        assert len(entity_docs) >= 3

        # Check function entity
        func_docs = [d for d in entity_docs if "calculate_sum" in d.metadata.additional_metadata.get("entity_name", "")]
        assert len(func_docs) == 1
        func_doc = func_docs[0]

        assert func_doc.metadata.additional_metadata["entity_type"] == "function"
        assert func_doc.metadata.additional_metadata["entity_name"] == "calculate_sum"
        assert "parameters" in func_doc.metadata.additional_metadata
        assert func_doc.metadata.additional_metadata["has_docstring"] == True
        assert func_doc.metadata.source_type == "codebase"
        assert func_doc.metadata.language == "python"

        # Check class entity
        class_docs = [d for d in entity_docs if d.metadata.additional_metadata.get("entity_name") == "Calculator"]
        assert len(class_docs) == 1
        class_doc = class_docs[0]

        assert class_doc.metadata.additional_metadata["entity_type"] == "class"
        assert class_doc.metadata.additional_metadata["entity_name"] == "Calculator"

        adapter.disconnect()

    def test_parse_javascript_file_with_entities(self, temp_repo):
        """Test parsing a JavaScript file into entity documents."""
        # Setup adapter
        config = {
            "repo_path": str(temp_repo),
            "languages": ["javascript"],
            "file_patterns": ["**/*.js"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        # Parse file
        entity_docs = adapter.parse_with_entities("helpers.js")

        # Should have entities: function + arrow function + class
        assert len(entity_docs) >= 3

        # Check function entity
        func_docs = [d for d in entity_docs if "greet" in d.metadata.additional_metadata.get("entity_name", "")]
        assert len(func_docs) == 1
        func_doc = func_docs[0]

        assert func_doc.metadata.additional_metadata["entity_type"] == "function"
        assert func_doc.metadata.additional_metadata["entity_name"] == "greet"
        assert func_doc.metadata.language == "javascript"

        # Check arrow function entity
        arrow_docs = [d for d in entity_docs if "add" in d.metadata.additional_metadata.get("entity_name", "")]
        assert len(arrow_docs) == 1
        arrow_doc = arrow_docs[0]

        assert arrow_doc.metadata.additional_metadata["entity_type"] == "function"
        assert arrow_doc.metadata.additional_metadata["entity_name"] == "add"

        # Check class entity
        class_docs = [d for d in entity_docs if d.metadata.additional_metadata.get("entity_name") == "Counter"]
        assert len(class_docs) == 1

        adapter.disconnect()

    def test_fetch_all_entities(self, temp_repo):
        """Test fetching all entities from repository."""
        # Setup adapter
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python", "javascript"],
            "file_patterns": ["**/*.py", "**/*.js"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        # Fetch all entities
        all_entities = list(adapter.fetch_all_entities())

        # Should have entities from both files
        assert len(all_entities) >= 6  # At least 3 from Python + 3 from JavaScript

        # Check we have both languages
        languages = {doc.metadata.language for doc in all_entities}
        assert "python" in languages
        assert "javascript" in languages

        # Check all have entity metadata
        for doc in all_entities:
            assert "entity_type" in doc.metadata.additional_metadata
            assert "entity_name" in doc.metadata.additional_metadata
            assert "file_path" in doc.metadata.additional_metadata

        adapter.disconnect()

    def test_entity_document_structure(self, temp_repo):
        """Test that entity documents have correct structure."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        entity_docs = adapter.parse_with_entities("utils.py")
        assert len(entity_docs) > 0

        # Check first entity document
        doc = entity_docs[0]

        # Check Document structure
        assert doc.content is not None
        assert len(doc.content) > 0
        assert doc.format == "markdown"

        # Check DocumentMetadata structure
        assert doc.metadata.source_id is not None
        assert doc.metadata.title is not None
        assert doc.metadata.source_type == "codebase"
        assert doc.metadata.last_modified is not None

        # Check additional metadata
        metadata = doc.metadata.additional_metadata
        assert "entity_type" in metadata
        assert "entity_name" in metadata
        assert "file_path" in metadata
        assert "programming_language" in metadata

        # Content should include code block
        assert "```" in doc.content
        assert "python" in doc.content.lower()

        adapter.disconnect()

    def test_entity_source_ids_are_unique(self, temp_repo):
        """Test that each entity gets a unique source ID."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        entity_docs = adapter.parse_with_entities("utils.py")

        # Collect all source IDs
        source_ids = [doc.metadata.source_id for doc in entity_docs]

        # Check all are unique
        assert len(source_ids) == len(set(source_ids))

        # Check format: should be file_path::entity_name or file_path::ClassName::method_name
        for source_id in source_ids:
            assert "::" in source_id
            assert "utils.py" in source_id

        adapter.disconnect()

    def test_parser_registry_lazy_loading(self):
        """Test that parser registry lazy-loads parsers."""
        registry = CodeParserRegistry()

        # Registry should have parser classes registered
        assert registry.supports_language("python")
        assert registry.supports_language("javascript")

        # But parsers should not be instantiated yet
        assert "python" not in registry._parsers
        assert "javascript" not in registry._parsers

        # Get parser - should instantiate
        python_parser = registry.get_parser("python")
        assert python_parser is not None
        assert "python" in registry._parsers

        # Getting again should return same instance
        python_parser2 = registry.get_parser("python")
        assert python_parser is python_parser2

    def test_parser_registry_file_detection(self):
        """Test parser registry detects language from file extension."""
        registry = CodeParserRegistry()

        # Test Python files
        assert registry.supports_file("test.py")
        assert registry.detect_language("test.py") == "python"

        # Test JavaScript files
        assert registry.supports_file("app.js")
        assert registry.detect_language("app.js") == "javascript"
        assert registry.supports_file("component.jsx")
        assert registry.detect_language("component.jsx") == "javascript"

        # Test TypeScript (uses JavaScript parser)
        assert registry.supports_file("app.ts")
        assert registry.detect_language("app.ts") == "typescript"

        # Test unsupported
        assert not registry.supports_file("README.md")
        assert registry.detect_language("README.md") is None

    def test_entity_formatter_preserves_metadata(self, temp_repo):
        """Test that formatter preserves all entity metadata."""
        from src.ingestion.parsers import PythonParser

        parser = PythonParser()
        formatter = CodeEntityFormatter()

        # Parse Python code
        code = '''
def test_function(x: int, y: int) -> int:
    """Test function with type hints."""
    return x + y
'''
        entities = parser.parse(code)
        assert len(entities) >= 1

        entity = entities[0]

        # Format entity
        doc = formatter.format_entity(entity, "test.py", "python")

        # Check all metadata preserved
        metadata = doc.metadata.additional_metadata
        assert metadata["entity_name"] == "test_function"
        assert metadata["entity_type"] == "function"
        assert "parameters" in metadata
        assert "return_type" in metadata
        assert metadata["return_type"] == "int"
        assert metadata["has_docstring"] is True

    def test_all_entities_have_unique_source_ids(self, temp_repo):
        """Test that all entities have properly formatted source IDs."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        entity_docs = adapter.parse_with_entities("utils.py")

        # Should have multiple entities
        assert len(entity_docs) >= 3

        # Check all source IDs are properly formatted
        for doc in entity_docs:
            source_id = doc.metadata.source_id
            assert "::" in source_id
            assert "utils.py" in source_id
            # Should have entity name after ::
            parts = source_id.split("::")
            assert len(parts) >= 2
            assert len(parts[-1]) > 0  # Entity name should not be empty

        adapter.disconnect()

    def test_empty_file_returns_no_entities(self, temp_repo):
        """Test that empty file returns no entities."""
        # Create empty file
        empty_file = temp_repo / "empty.py"
        empty_file.write_text("")

        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        entity_docs = adapter.parse_with_entities("empty.py")

        # Empty file should return no entities
        assert len(entity_docs) == 0

        adapter.disconnect()

    def test_file_with_syntax_error_raises_exception(self, temp_repo):
        """Test that file with syntax errors raises exception."""
        # Create file with syntax error
        bad_file = temp_repo / "bad.py"
        bad_file.write_text("def broken(:\n    pass")

        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"]
        }
        adapter = CodebaseAdapter(config)
        adapter.connect()

        # Should raise IngestionError
        from src.ingestion.adapters.base import IngestionError
        with pytest.raises(IngestionError):
            adapter.parse_with_entities("bad.py")

        adapter.disconnect()
