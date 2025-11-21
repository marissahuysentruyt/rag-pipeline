"""
Tests for CodebaseAdapter.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.ingestion.adapters import CodebaseAdapter, ConfigurationError


class TestCodebaseAdapter:
    """Test suite for CodebaseAdapter."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository with sample files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create sample Python file
            python_file = repo_path / "example.py"
            python_file.write_text("""
def hello_world():
    '''Say hello to the world.'''
    print("Hello, World!")

class Calculator:
    '''A simple calculator class.'''
    
    def add(self, a, b):
        '''Add two numbers.'''
        return a + b
""")
            
            # Create sample JavaScript file
            js_file = repo_path / "example.js"
            js_file.write_text("""
function greet(name) {
    return `Hello, ${name}!`;
}

class Counter {
    constructor() {
        this.count = 0;
    }
    
    increment() {
        this.count++;
    }
}
""")
            
            # Create a node_modules directory (should be excluded)
            node_modules = repo_path / "node_modules"
            node_modules.mkdir()
            (node_modules / "package.js").write_text("// Should be excluded")
            
            yield repo_path
    
    def test_adapter_initialization(self, temp_repo):
        """Test adapter initialization with valid config."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python", "javascript"]
        }
        
        adapter = CodebaseAdapter(config)
        assert adapter.repo_path == temp_repo
        assert "python" in adapter.languages
        assert "javascript" in adapter.languages
    
    def test_adapter_invalid_path(self):
        """Test adapter with invalid repository path."""
        config = {
            "repo_path": "/nonexistent/path"
        }
        
        with pytest.raises(ConfigurationError):
            CodebaseAdapter(config)
    
    def test_list_documents(self, temp_repo):
        """Test listing documents from repository."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python", "javascript"]
        }
        
        adapter = CodebaseAdapter(config)
        adapter.connect()
        
        documents = adapter.list_documents()
        
        # Should find Python and JavaScript files, but not node_modules
        assert len(documents) == 2
        assert any("example.py" in doc for doc in documents)
        assert any("example.js" in doc for doc in documents)
        assert not any("node_modules" in doc for doc in documents)
        
        adapter.disconnect()
    
    def test_fetch_document(self, temp_repo):
        """Test fetching a single document."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python"]
        }
        
        adapter = CodebaseAdapter(config)
        adapter.connect()
        
        document = adapter.fetch_document("example.py")
        
        assert document is not None
        assert document.metadata.source_type == "codebase"
        assert document.metadata.title == "example.py"
        assert "hello_world" in document.content
        assert document.format == "markdown"
        
        adapter.disconnect()
    
    def test_fetch_all(self, temp_repo):
        """Test fetching all documents."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python", "javascript"]
        }
        
        adapter = CodebaseAdapter(config)
        adapter.connect()
        
        documents = list(adapter.fetch_all())
        
        assert len(documents) == 2
        assert all(doc.metadata.source_type == "codebase" for doc in documents)
        
        adapter.disconnect()
    
    def test_get_stats(self, temp_repo):
        """Test getting repository statistics."""
        config = {
            "repo_path": str(temp_repo),
            "languages": ["python", "javascript"]
        }
        
        adapter = CodebaseAdapter(config)
        adapter.connect()
        
        stats = adapter.get_stats()
        
        assert stats["total_files"] == 2
        assert "python" in stats["languages"]
        assert "javascript" in stats["languages"]
        assert stats["total_size_bytes"] > 0
        
        adapter.disconnect()

