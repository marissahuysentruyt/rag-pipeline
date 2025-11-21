# RAG Pipeline for Design System Documentation

A Retrieval-Augmented Generation (RAG) pipeline built with Haystack and Anthropic Claude to enhance search and discovery within design system documentation.

## Features

- Semantic search across design system documentation
- Component-aware chunking and indexing
- Context-aware responses using Claude 3.5 Sonnet
- Support for multiple documentation formats (Markdown, MDX, HTML, JSON)
- Docker-based deployment for easy setup

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Anthropic API key

### Setup

1. Clone the repository and navigate to the project directory

2. Copy the environment template and add your API keys:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

3. Start the services with Docker Compose:
```bash
docker-compose up -d
```

4. The API will be available at `http://localhost:8000`

### Local Development (without Docker)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. Verify your setup:
```bash
python tests/test_setup.py
```

5. Run the demo:
```bash
python demo_simple_rag.py
```

See [tests/TESTING.md](tests/TESTING.md) for detailed testing instructions and troubleshooting.

## Project Structure

```
rag-pipeline/
├── src/
│   ├── ingestion/          # Document ingestion
│   │   ├── adapters/       # Ingestion source adapters (web, file, CMS, DB)
│   │   ├── crawl_docs.py   # Web crawler CLI
│   │   ├── web_crawler.py  # Incremental web crawler
│   │   ├── document_processor.py  # Markdown parsing & chunking
│   │   └── document_indexer.py    # Embedding & indexing
│   │
│   ├── processing/         # Document processing
│   │   └── chunkers/       # Chunking strategies (markdown, semantic, fixed-size)
│   │
│   ├── embedding/          # Embedding generation
│   │   └── providers/      # Embedding providers (sentence-transformers, OpenAI)
│   │
│   ├── retrieval/          # Document retrieval
│   │   └── strategies/     # Retrieval strategies (vector, hybrid, BM25)
│   │
│   ├── generation/         # LLM integration
│   │   └── providers/      # LLM providers (Anthropic, OpenAI, local)
│   │
│   ├── query/              # Query interface
│   │   ├── interfaces/     # Query interfaces (CLI, API, Custom GPT)
│   │   └── rag_pipeline.py # RAG orchestration
│   │
│   └── api/               # REST API
│       └── server.py      # FastAPI server
│
├── data/
│   ├── raw/crawled/       # Crawled markdown files
│   └── chroma_db/         # Chroma vector database (2,147 docs)
│
├── tests/                 # 71 unit tests
├── config/                # Configuration files
├── query.py               # CLI query tool
└── rag-plan.md           # Detailed implementation plan
```

## Architecture

This project uses a **modular adapter-based architecture** to support multiple data sources, embedding providers, LLMs, and query interfaces.

### Core Abstractions

**1. Ingestion Adapters** (`src/ingestion/adapters/`)
- Abstract base class for different document sources
- Current: Web crawler, Codebase adapter (11+ languages)
- Planned: File system, CMS (Contentful), Database readers

**2. Chunking Strategies** (`src/processing/chunkers/`)
- Abstract base class for text chunking approaches
- Current: Markdown chunker, Code chunker (language-aware, preserves entities)
- Planned: Semantic chunking, fixed-size chunking

**3. Embedding Providers** (`src/embedding/providers/`)
- Abstract base class for embedding models
- Current: Sentence Transformers (all-MiniLM-L6-v2, local), OpenAI (API)
- Planned: Cohere embeddings, custom models

**4. Retrieval Strategies** (`src/retrieval/strategies/`)
- Abstract base class for document retrieval
- Current: ChromaRetriever (vector similarity, metadata filtering)
- Planned: Hybrid search, BM25, reranking

**5. LLM Providers** (`src/generation/providers/`)
- Abstract base class for language models
- Current: Anthropic Claude (Sonnet 4.5, streaming support)
- Planned: OpenAI GPT-4, local models (Ollama)

**6. Query Interfaces** (`src/query/interfaces/`)
- Abstract base class for query endpoints
- Current: CLI (Rich), REST API (FastAPI)
- Planned: Custom GPT Actions, OpenAI-compatible API

### Benefits of Modular Design

- **Extensibility**: Easy to add new providers without changing core logic
- **Testability**: Each component can be tested independently (159 tests, all passing)
- **Maintainability**: Clear separation of concerns
- **Flexibility**: Swap implementations without breaking existing code

See `rag-plan.md` for detailed implementation plan and architecture evolution.

## Codebase Ingestion

In addition to documentation, you can index source code from local repositories to enable queries about actual implementations.

### Quick Start

```python
from src.ingestion.adapters import CodebaseAdapter
from src.ingestion.parsers import PythonParser, JavaScriptParser

# Configure adapter for your codebase
config = {
    "repo_path": "/path/to/your/codebase",
    "languages": ["python", "javascript", "typescript"],
    "file_patterns": ["**/*.py", "**/*.js", "**/*.ts"],
    "exclude_patterns": ["**/node_modules/**", "**/dist/**"]
}

# Ingest codebase
adapter = CodebaseAdapter(config)
adapter.connect()

for doc in adapter.fetch_all():
    print(f"File: {doc.metadata.title} ({doc.metadata.language})")

# Parse individual files for entity extraction
parser = PythonParser()
entities = parser.parse(source_code)
for entity in entities:
    print(f"{entity.entity_type.value}: {entity.name} - {entity.signature}")
```

### Supported Languages

- ✅ Python (AST-based parsing)
- ✅ JavaScript/TypeScript (regex-based parsing)
- 🔜 Additional parsers planned (Java, Go, Rust)

### Features

- **Smart Chunking**: Code chunker preserves complete functions and classes
- **Rich Metadata**: Extracts signatures, parameters, return types, docstrings
- **Incremental Updates**: Only re-index modified files
- **Multi-language Support**: 11+ languages with extensible parser system

See [Code Parsing Tests](tests/test_code_parsers.py) for usage examples.

## Development

- Run tests: `pytest tests/`
- Format code: `black src/`
- Lint: `ruff check src/`

## Using the CLI Query Tool

The project includes a rich command-line interface for querying the indexed documentation.

### Basic Usage

**Single query:**
```bash
python query.py "How do I use a Button component in React Spectrum?"
```

**Interactive mode:**
```bash
python query.py --interactive
```

### Interactive Commands

Once in interactive mode, you can use these commands:
- `help` - Show available commands
- `sources on/off` - Toggle source citations display
- `clear` - Clear the screen
- `exit`, `quit`, `q` - Exit the program

### Query Options

**Filter by domain:**
```bash
python query.py "color guidelines" --domain "spectrum.adobe.com"
```

**Adjust number of results:**
```bash
python query.py "Button examples" --top-k 10
```

**Hide source citations:**
```bash
python query.py "What is Spectrum?" --no-sources
```

**Enable verbose logging:**
```bash
python query.py "button" --verbose
```

### Example Session

```bash
$ python query.py --interactive

╭────────────────────────────────────╮
│ Design System Documentation Search │
│ Powered by RAG + Claude 4.5        │
╰────────────────────────────────────╯

Type your questions below. Type 'exit' or 'quit' to leave.
Type 'help' for available commands.

❯ How do I use a button in React Spectrum?
[Returns formatted answer with code examples and source citations]

❯ sources off
Source display disabled

❯ What are the color tokens?
[Returns answer without showing sources]

❯ exit
Goodbye!
```

### Example Output

When you run a query, you'll see:
1. **Question** - Your query displayed clearly
2. **Answer** - Formatted markdown response from Claude with code examples
3. **Sources** - Table showing retrieved documents with titles, URLs, and relevance scores

```
╭─────────────────────────────────── Answer ───────────────────────────────────╮
│ To use a Button component in React Spectrum:                                 │
│                                                                              │
│ 1. Install the package:                                                      │
│    npm install @adobe/react-spectrum                                         │
│                                                                              │
│ 2. Import and use:                                                           │
│    import { Button } from '@adobe/react-spectrum';                           │
│                                                                              │
│    <Button variant="accent">Click me</Button>                                │
╰──────────────────────────────────────────────────────────────────────────────╯

                                    Sources
┏━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ # ┃ Title               ┃ URL                     ┃ Score ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 1 │ Button – React      │ https://react-spectr... │ 0.599 │
│ 2 │ Provider – React    │ https://react-spectr... │ 0.748 │
│ 3 │ ButtonGroup         │ https://react-spectr... │ 0.825 │
└───┴─────────────────────┴─────────────────────────┴───────┘
```

## Running Tests

The project includes comprehensive test coverage with 71 unit tests.

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run all tests with summary
pytest tests/ -q

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Suites

```bash
# Document processing tests (20 tests)
pytest tests/test_document_processor.py -v

# Document indexing tests (25 tests)
pytest tests/test_document_indexer.py -v

# RAG pipeline tests (21 tests)
pytest tests/test_rag_pipeline.py -v

# Setup verification tests (5 tests)
pytest tests/test_setup.py -v
```

### Test Coverage Breakdown

**Total: 71 tests**

**Document Processor (20 tests)**
- Markdown parsing with YAML frontmatter
- Code block extraction using `[code]...[/code]` patterns
- Section extraction by markdown headings
- Intelligent chunking (200-1500 chars)
- Metadata preservation
- File and directory processing

**Document Indexer (25 tests)**
- ChromaDB integration
- Embedding generation (384-dimensional vectors)
- Batch processing with configurable sizes
- Collection management and persistence
- Duplicate handling policies
- Edge cases (empty content, long content, special chars)

**RAG Pipeline (21 tests)**
- Pipeline initialization and configuration
- Semantic document retrieval
- Metadata filtering (domain, title, etc.)
- Query generation with mocked Claude responses
- Prompt building and component connections
- Error handling and propagation
- Multiple queries on same instance

**Setup Verification (5 tests)**
- Dependency availability checks
- API key validation
- ChromaDB connectivity
- Haystack pipeline functionality
- Anthropic integration

### Example Test Output

```bash
$ pytest tests/ -v

tests/test_document_processor.py::TestMarkdownParsing::test_parse_file_with_frontmatter PASSED
tests/test_document_processor.py::TestCodeBlockExtraction::test_extract_code_blocks PASSED
tests/test_document_indexer.py::TestIndexing::test_index_multiple_chunks PASSED
tests/test_rag_pipeline.py::TestQueryGeneration::test_query_basic PASSED

======================== 71 passed in 9.49s ========================
```

### Continuous Testing During Development

For active development, automatically run tests on file changes:

```bash
# Install pytest-watch
pip install pytest-watch

# Run in watch mode
ptw tests/
```

## Building and Updating the Index

### Current Index Status

The system currently has:
- **2,147 indexed document chunks**
- **314 source pages** from Adobe Spectrum documentation
- **37 MB** vector database
- **3 documentation sources**: Spectrum Web Components, React Spectrum, Spectrum Design System

### Crawling Documentation

To update or rebuild the documentation index:

**1. Crawl configured sources:**
```bash
python src/ingestion/crawl_docs.py crawl
```

**2. Add a new documentation source:**
```bash
python src/ingestion/crawl_docs.py add-source \
  --name "My Design System" \
  --url "https://example.com/docs" \
  --max-depth 3
```

**3. List all configured sources:**
```bash
python src/ingestion/crawl_docs.py list-sources
```

**4. Remove a source:**
```bash
python src/ingestion/crawl_docs.py remove-source "My Design System"
```

Crawler configuration is stored in `config/crawler_config.yaml`.

### Indexing Documents

After crawling, index the documents into the vector database:

```bash
python src/ingestion/document_indexer.py
```

This process:
1. Reads markdown files from `data/raw/crawled/`
2. Parses YAML frontmatter and content
3. Creates intelligent chunks (preserves code blocks)
4. Generates embeddings using sentence-transformers
5. Stores in ChromaDB at `data/chroma_db/`

**Expected output:**
```
2025-11-18 12:00:00 - Processing documents...
2025-11-18 12:00:05 - Created 2,161 chunks from 314 files
2025-11-18 12:00:10 - Indexing documents...
2025-11-18 12:00:53 - Successfully indexed 2,147 documents
2025-11-18 12:00:53 - Stats: {'total_documents': 2147, 'collection_name': 'design_system_docs'}
```

## REST API

The project includes a FastAPI-based REST API for querying the documentation.

### Starting the API Server

**Production mode:**
```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

**Development mode (with auto-reload):**
```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### GET /
Get API information and available endpoints.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
    "name": "Design System RAG API",
    "version": "1.0.0",
    "endpoints": {
        "/query": "POST - Query the documentation",
        "/health": "GET - Health check",
        "/stats": "GET - Index statistics",
        "/refresh": "POST - Refresh documentation index"
    }
}
```

#### GET /health
Health check endpoint to verify API status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "database_connected": true,
    "api_key_configured": true,
    "total_documents": 2147
}
```

#### GET /stats
Get statistics about the indexed documents.

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
    "total_documents": 2147,
    "collection_name": "design_system_docs",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "claude-sonnet-4-5-20250929",
    "persist_path": "./data/chroma_db",
    "last_refresh": null
}
```

#### POST /query
Query the design system documentation using RAG.

**Request body:**
```json
{
    "question": "How do I use a Button component?",
    "domain": "react-spectrum.adobe.com",  // Optional: filter by domain
    "top_k": 5  // Optional: number of documents to retrieve (default: 5, max: 20)
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I use a Button component?",
    "top_k": 3
  }'
```

**Response:**
```json
{
    "answer": "# How to Use a Button Component\n\nThe approach depends on which...",
    "sources": [
        {
            "title": "Button – React Spectrum",
            "url": "https://react-spectrum.adobe.com/react-spectrum/Button.html",
            "score": 0.824,
            "content": "# Button\n\nButtons allow users to perform..."
        }
    ],
    "metadata": {
        "query": "How do I use a Button component?",
        "num_documents": 3,
        "model": "claude-sonnet-4-5-20250929",
        "filters": null
    }
}
```

**Query with domain filter:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "color tokens",
    "domain": "spectrum.adobe.com",
    "top_k": 3
  }'
```

#### POST /refresh
Trigger a documentation refresh (recrawl and re-index). This is an asynchronous background task.

```bash
curl -X POST http://localhost:8000/refresh
```

**Response:**
```json
{
    "message": "Documentation refresh started",
    "status": "started",
    "timestamp": "2025-11-18T20:00:00.000Z"
}
```

Check the `/stats` endpoint to see when the refresh completed by checking the `last_refresh` field.

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These interfaces allow you to explore and test all endpoints directly from your browser.

### API Error Handling

The API returns standard HTTP status codes:

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service not ready (e.g., during startup)

**Error response format:**
```json
{
    "detail": "Error message describing what went wrong"
}
```

## Vector Database

Currently using **Chroma** for vector storage and retrieval. Chroma provides a good balance of ease-of-use, Docker compatibility, and features suitable for design system documentation.

**TODO:** Explore **Qdrant** for production deployments. Qdrant offers:
- Superior performance for large-scale vector search
- Advanced metadata filtering capabilities
- Better horizontal scaling options
- More granular control over indexing strategies

## License

[Your License Here]
