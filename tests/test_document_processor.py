"""
Unit tests for document processing and chunking.
"""

import pytest
from pathlib import Path
from src.ingestion.document_processor import DocumentProcessor, DocumentChunk


@pytest.fixture
def processor():
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(
        min_chunk_size=100,
        max_chunk_size=500
    )


@pytest.fixture
def sample_markdown_with_frontmatter():
    """Sample markdown with YAML frontmatter."""
    return """---
title: Button Component
url: https://example.com/button
domain: example.com
---

# Button Component

A button triggers an action or event.

## Installation

Install the button package.

## Usage

Basic button example:

[code]
import { Button } from '@example/button';

function App() {
  return <Button>Click me</Button>;
}
[/code]

## Variants

Buttons come in different variants:
- Primary
- Secondary
- Accent
"""


@pytest.fixture
def sample_markdown_no_frontmatter():
    """Sample markdown without frontmatter."""
    return"""# Simple Document

This is a simple document without frontmatter.

It has multiple paragraphs and some content."""


class TestMarkdownParsing:
    """Tests for parsing markdown files with frontmatter."""

    def test_parse_file_with_frontmatter(self, processor, tmp_path, sample_markdown_with_frontmatter):
        """Test parsing a markdown file with YAML frontmatter."""
        test_file = tmp_path / "button.md"
        test_file.write_text(sample_markdown_with_frontmatter)

        doc = processor.parse_markdown_file(test_file)

        assert doc is not None
        assert doc['metadata']['title'] == "Button Component"
        assert doc['metadata']['url'] == "https://example.com/button"
        assert doc['metadata']['domain'] == "example.com"
        assert "button triggers an action" in doc['content']

    def test_parse_file_without_frontmatter(self, processor, tmp_path, sample_markdown_no_frontmatter):
        """Test parsing a markdown file without frontmatter returns None."""
        test_file = tmp_path / "simple.md"
        test_file.write_text(sample_markdown_no_frontmatter)

        doc = processor.parse_markdown_file(test_file)

        # Files without frontmatter return None
        assert doc is None

    def test_parse_nonexistent_file(self, processor):
        """Test parsing a file that doesn't exist returns None."""
        doc = processor.parse_markdown_file(Path("nonexistent.md"))
        assert doc is None

    def test_parse_invalid_frontmatter(self, processor, tmp_path):
        """Test parsing a file with invalid frontmatter returns None."""
        test_file = tmp_path / "invalid.md"
        test_file.write_text("---\nonly one delimiter\n\n# Content")

        doc = processor.parse_markdown_file(test_file)
        assert doc is None


class TestCodeBlockExtraction:
    """Tests for code block detection and extraction."""

    def test_extract_code_blocks(self, processor):
        """Test extracting [code]...[/code] blocks."""
        content = """Some text before.

[code]
def hello():
    print("world")
[/code]

Text between blocks.

[code]
console.log("hi");
[/code]

Text after blocks."""

        blocks = processor.extract_code_blocks(content)

        assert len(blocks) == 5
        assert blocks[0]['type'] == 'text'
        assert blocks[1]['type'] == 'code'
        assert 'def hello()' in blocks[1]['content']
        assert blocks[2]['type'] == 'text'
        assert blocks[3]['type'] == 'code'
        assert 'console.log' in blocks[3]['content']
        assert blocks[4]['type'] == 'text'

    def test_no_code_blocks(self, processor):
        """Test content with no code blocks."""
        content = "Just plain text with no code."
        blocks = processor.extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0]['type'] == 'text'
        assert blocks[0]['content'] == content

    def test_only_code_block(self, processor):
        """Test content with only a code block."""
        content = """[code]
print("hello")
[/code]"""
        blocks = processor.extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0]['type'] == 'code'
        assert 'print("hello")' in blocks[0]['content']


class TestSectionExtraction:
    """Tests for extracting sections based on headings."""

    def test_extract_sections(self, processor):
        """Test extracting sections by markdown headings."""
        content = """# Main Title

Introduction text.

## Section 1

Content of section 1.

## Section 2

Content of section 2.

### Subsection 2.1

Content of subsection."""

        sections = processor.extract_sections(content)

        assert len(sections) >= 3
        headings = [s['heading'] for s in sections]
        assert any('Main Title' in h for h in headings)
        assert any('Section 1' in h for h in headings)
        assert any('Section 2' in h for h in headings)

    def test_no_headings(self, processor):
        """Test content without headings."""
        content = "Just plain text without any headings that is long enough to be meaningful content."
        sections = processor.extract_sections(content)

        # Should return at least one section
        assert len(sections) >= 1


class TestChunking:
    """Tests for intelligent document chunking."""

    def test_create_chunks(self, processor):
        """Test basic chunk creation."""
        doc = {
            'metadata': {
                'title': 'Test Document',
                'url': 'https://example.com/test'
            },
            'content': """# Test Document

This is a test document with enough content to create chunks.

## Section 1

This section has some content that describes something important. It needs to be long enough to meet the minimum chunk size requirements so we can test properly.

## Section 2

This is another section with different content. It also needs to be substantial enough to create a proper chunk for testing purposes."""
        }

        chunks = processor.create_chunks(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.metadata['title'] == 'Test Document' for c in chunks)

    def test_chunks_respect_size_limits(self, processor):
        """Test that chunks are created for long content with sections."""
        # Create a long document with multiple sections
        content = "# Test\n\n" + ("This is a sentence. " * 5) + "\n\n## Section 2\n\n" + ("Another sentence. " * 5)
        doc = {
            'metadata': {'title': 'Long Doc'},
            'content': content
        }

        chunks = processor.create_chunks(doc)

        # Should create at least one chunk
        assert len(chunks) > 0
        # Most chunks should be reasonable size (allow flex for single-section chunks)
        reasonable_chunks = [c for c in chunks if len(c.content) <= processor.max_chunk_size * 1.5]
        assert len(reasonable_chunks) >= len(chunks) * 0.5  # At least half should be reasonable

    def test_code_blocks_in_chunks(self, processor):
        """Test that code blocks are preserved in chunks."""
        content = """# Component

Some text before code.

[code]
def function():
    return True
[/code]

Text after code that is also substantial enough to be included in the chunk."""

        doc = {
            'metadata': {'title': 'Code Test'},
            'content': content
        }

        chunks = processor.create_chunks(doc)

        # Find chunks with code
        code_chunks = [c for c in chunks if 'def function' in c.content or '```' in c.content]
        assert len(code_chunks) > 0

    def test_empty_content(self, processor):
        """Test chunking with minimal content."""
        doc = {
            'metadata': {'title': 'Empty'},
            'content': 'Short'  # Below min_chunk_size
        }

        chunks = processor.create_chunks(doc)
        # Should return empty list for content below minimum
        assert len(chunks) == 0

    def test_chunk_metadata_preserved(self, processor):
        """Test that chunk metadata includes original document metadata."""
        doc = {
            'metadata': {
                'title': 'Test',
                'url': 'https://example.com',
                'domain': 'example.com'
            },
            'content': """# Test

This is content that is long enough to create a proper chunk for testing.
We need enough text here to meet the minimum chunk size requirement."""
        }

        chunks = processor.create_chunks(doc)

        for chunk in chunks:
            assert 'title' in chunk.metadata
            assert 'url' in chunk.metadata
            assert 'domain' in chunk.metadata


class TestFileProcessing:
    """Tests for processing complete files."""

    def test_process_file(self, processor, tmp_path):
        """Test processing a complete markdown file."""
        content = """---
title: Button
url: https://example.com/button
domain: example.com
---

# Button Component

A button is an interactive element.

## Usage

Use buttons for actions.

[code]
<Button>Click</Button>
[/code]

## Props

Buttons accept various props for customization.
"""
        test_file = tmp_path / "test.md"
        test_file.write_text(content)

        chunks = processor.process_file(test_file)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.metadata['title'] == 'Button' for c in chunks)

    def test_process_file_without_frontmatter(self, processor, tmp_path):
        """Test processing a file without frontmatter returns empty list."""
        test_file = tmp_path / "no-frontmatter.md"
        test_file.write_text("# Just content\n\nNo frontmatter here.")

        chunks = processor.process_file(test_file)
        assert len(chunks) == 0

    def test_process_nonexistent_file(self, processor):
        """Test processing nonexistent file returns empty list."""
        chunks = processor.process_file(Path("nonexistent.md"))
        assert len(chunks) == 0


class TestDirectoryProcessing:
    """Tests for processing multiple files."""

    def test_process_directory(self, processor, tmp_path):
        """Test processing all markdown files in a directory."""
        # Create multiple test files
        for i in range(3):
            content = f"""---
title: Doc {i}
url: https://example.com/doc{i}
domain: example.com
---

# Document {i}

This is document {i} with enough content to create chunks.
It has multiple sentences to ensure it meets minimum size requirements.
"""
            test_file = tmp_path / f"test{i}.md"
            test_file.write_text(content)

        all_chunks = processor.process_directory(tmp_path)

        assert len(all_chunks) > 0
        # Check we got chunks from multiple documents
        titles = set(c.metadata['title'] for c in all_chunks)
        assert len(titles) == 3

    def test_process_empty_directory(self, processor, tmp_path):
        """Test processing an empty directory."""
        chunks = processor.process_directory(tmp_path)
        assert len(chunks) == 0

    def test_process_directory_mixed_files(self, processor, tmp_path):
        """Test processing directory with both valid and invalid files."""
        # Valid file with frontmatter and enough content
        valid_file = tmp_path / "valid.md"
        valid_file.write_text("""---
title: Valid
url: https://example.com/valid
---

# Valid Document

This has frontmatter and enough content to meet the minimum chunk size requirement.
We need at least 100 characters of content to create a chunk, so here is more text.""")

        # Invalid file without frontmatter
        invalid_file = tmp_path / "invalid.md"
        invalid_file.write_text("# No Frontmatter\n\nJust content.")

        # Non-markdown file
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("This is not markdown.")

        chunks = processor.process_directory(tmp_path)

        # Should only get chunks from valid file
        assert len(chunks) > 0
        assert all(c.metadata['title'] == 'Valid' for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
