"""
Tests for CodeChunker.
"""

import pytest
from src.processing.chunkers import CodeChunker, ChunkingConfig, ChunkType


class TestCodeChunker:
    """Test suite for CodeChunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create a CodeChunker instance."""
        config = ChunkingConfig(
            min_chunk_size=50,
            max_chunk_size=1000,
            preserve_code_blocks=True
        )
        return CodeChunker(config)
    
    def test_chunk_markdown_code(self, chunker):
        """Test chunking markdown-formatted code."""
        text = """# Example Function

**File:** `example.py`
**Language:** python

```python
def hello_world():
    '''Say hello.'''
    print("Hello, World!")
```
"""
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert chunks[0].chunk_type == ChunkType.CODE
        assert "hello_world" in chunks[0].content
    
    def test_chunk_python_code(self, chunker):
        """Test chunking raw Python code."""
        text = """
def function_one():
    '''First function.'''
    return 1

def function_two():
    '''Second function.'''
    return 2

class MyClass:
    '''A simple class.'''
    
    def method_one(self):
        return "method"
"""
        
        metadata = {"programming_language": "python"}
        chunks = chunker.chunk_text(text, metadata)
        
        # Should create multiple chunks for different entities
        assert len(chunks) > 0
        assert all(chunk.chunk_type == ChunkType.CODE for chunk in chunks)
    
    def test_chunk_javascript_code(self, chunker):
        """Test chunking JavaScript code."""
        text = """
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
"""
        
        metadata = {"programming_language": "javascript"}
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        assert all(chunk.chunk_type == ChunkType.CODE for chunk in chunks)
    
    def test_detect_language_from_extension(self, chunker):
        """Test language detection from file extension."""
        text = "def test(): pass"
        metadata = {"file_extension": ".py"}
        
        language = chunker._detect_language(text, metadata)
        assert language == "python"
    
    def test_detect_language_from_content(self, chunker):
        """Test language detection from content patterns."""
        python_text = "def hello():\n    pass"
        language = chunker._detect_language(python_text, {})
        assert language == "python"
        
        js_text = "function hello() { return true; }"
        language = chunker._detect_language(js_text, {})
        assert language == "javascript"
    
    def test_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
        
        chunks = chunker.chunk_text("   \n  \n  ")
        assert len(chunks) == 0
    
    def test_chunk_with_metadata(self, chunker):
        """Test that metadata is preserved in chunks."""
        text = "def test(): pass"
        metadata = {
            "file_path": "test.py",
            "programming_language": "python"
        }
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        assert chunks[0].metadata == metadata

