# RAG Pipeline Plan for Design System Documentation Search

## Overview
Build a RAG (Retrieval-Augmented Generation) pipeline to enhance how users can search through design system documentation.

## Phase 1: Foundation & Setup

### Python Environment Setup

**Python Version & Dependency Management**
- Python 3.11+ (for better performance and type hints)
- Poetry or pip with `requirements.txt` for dependency management
- Virtual environment isolation (handled by Docker)

**Project Structure**
```
rag-pipeline/
├── src/
│   ├── ingestion/       # Document loaders and parsers
│   ├── processing/      # Chunking and preprocessing
│   ├── embedding/       # Embedding generation
│   ├── retrieval/       # Vector search and ranking
│   ├── generation/      # LLM integration
│   └── api/            # Query interface
├── data/
│   ├── raw/            # Original documentation files
│   ├── processed/      # Chunked and enriched data
│   └── vectorstore/    # Vector database persistence
├── tests/
├── config/             # Configuration files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt    # or pyproject.toml
└── .env.example       # Environment variables template
```

**Docker Container Setup**
- Dockerfile for the RAG application
- Multi-stage build for smaller image size
- Docker Compose for orchestration (app + vector DB + optional services)
- Volume mounts for data persistence and development
- Environment variables for API keys and configuration

**Core Dependencies**
- **RAG Framework**: Haystack (`haystack-ai`)
  - Modern, production-ready framework from deepset
  - Pipeline-based architecture for flexibility
  - Excellent documentation and component ecosystem
  - Native support for multiple vector databases and LLMs
- **LLM Integration**: Anthropic Claude (`anthropic-haystack`)
  - Official Haystack integration for Anthropic's Claude models
  - Supports Claude 3.5 Sonnet and other Claude models
  - Streaming and non-streaming response support
- **Vector Database Options**:
  - FAISS: Local, fast, good for development (no separate service needed)
  - Chroma: Docker-compatible, persistent, good balance
  - Weaviate: Production-grade, runs in Docker
  - Pinecone: Cloud-hosted (no Docker needed, requires API)
  - Qdrant: High-performance, Docker-compatible
- **Embedding Model Provider**: OpenAI, Cohere, or open-source (sentence-transformers)

**Docker Compose Services**
```yaml
services:
  rag-app:          # Main application
  chroma:           # Vector database (if using Chroma)
  # postgres:       # Optional: metadata storage
  # redis:          # Optional: caching layer
```

**Development vs Production Considerations**
- Development: Use FAISS or Chroma locally, mount source code for hot-reload
- Production: Use persistent volumes, health checks, resource limits

## Phase 2: Documentation Processing

### Phase 2a: Web Crawling & Collection
**Incremental web crawler for documentation sites**
- Crawl multiple documentation sources (public websites)
- Change detection using content hashing
- Respect robots.txt and rate limiting
- Extract main content from HTML using readability
- Convert HTML to clean Markdown
- Store crawl state for incremental updates
- Force re-crawl after configurable interval (default: 7 days)

**Content extraction and cleaning**
- Remove navigation, headers, footers, sidebars
- Preserve code blocks, tables, and important formatting
- Extract metadata from HTML (title, meta tags, URL structure)
- Custom metadata extractors (CSS selectors for component names, categories)
- Minimum content length filtering

### Phase 2b: Documentation Ingestion Pipeline
**Support multiple formats**
- Markdown (from crawled sites or local files)
- MDX (Markdown with JSX components)
- HTML (pre-crawled or local)
- JSON (design tokens, component metadata)

**Extract component metadata**
- Component props, variants, usage examples
- Preserve code snippets and their language context
- Handle images/diagrams (descriptions or multimodal embeddings)

**Implement intelligent chunking strategy**
- Keep component docs together (don't split mid-component)
- Preserve hierarchical structure (category → component → props)
- Maintain metadata: component name, category, tags, version
- Special handling for code examples vs. descriptive text

## Phase 3: Indexing & Retrieval
**Set up embeddings and vector storage**
- Generate embeddings for each documentation chunk
- Store with rich metadata for filtering
- Create indexes for efficient similarity search

**Build semantic search system**
- Implement hybrid search (semantic + keyword for code snippets)
- Add metadata filters (component type, category, framework version)
- Ranking algorithm tuned for design system queries

## Phase 4: Generation & Interface
**Integrate LLM for response generation**
- Craft prompts that understand design system context
- Include retrieved chunks as context
- Format responses with code examples and links

**Create query interface**
- REST API or CLI for querying
- Support for follow-up questions
- Response formatting (markdown, JSON)

## Phase 5: Quality & Production
**Build evaluation framework**
- Test queries: "How do I use Button with icons?", "What are the spacing tokens?"
- Measure retrieval accuracy and response quality
- Iterate on chunking and retrieval strategies

**Add production features**
- Logging and monitoring
- Rate limiting
- Cache frequently asked questions
- Version control for documentation updates

## Key Considerations for Design Systems

### Component-centric Architecture
Structure retrieval around components, not just text similarity. Each component should be a logical unit with its props, examples, and guidelines.

### Code-aware Processing
Preserve syntax and framework context (React vs Vue vs Angular). Code snippets should be searchable both semantically and via exact matches.

### Visual Context
Handle design tokens, color palettes, spacing scales. These structured data elements need special indexing.

### Version Management
Support multiple documentation versions as design systems evolve.

### Cross-references
Maintain links between related components (e.g., Button → Icon, Form → Input).

## Implementation Status

### ✅ Phase 1: Foundation & Setup (Completed)
- ✅ Created project structure with src/, data/, tests/, config/
- ✅ Set up virtual environment with Python 3.13
- ✅ Installed Haystack AI framework (`haystack-ai>=2.0.0`)
- ✅ Integrated Anthropic Claude via `anthropic-haystack`
- ✅ Configured Chroma as vector database with `chroma-haystack`
- ✅ Set up sentence-transformers for embeddings (`all-MiniLM-L6-v2`)
- ✅ Created Dockerfile and docker-compose.yml for future deployment
- ✅ Tested basic RAG demo with Claude Sonnet 4.5

**Files Created:**
- `requirements.txt`, `.env.example`, `Dockerfile`, `docker-compose.yml`
- `tests/test_setup.py`, `demo_simple_rag.py`

### ✅ Phase 2a: Web Crawling (Completed)
- ✅ Implemented incremental web crawler (`web_crawler.py`)
- ✅ Added change detection using SHA256 content hashing
- ✅ Integrated Playwright for JavaScript-rendered sites
- ✅ HTML to Markdown conversion with html2text
- ✅ Content extraction with readability-lxml
- ✅ Crawl state persistence in SQLite
- ✅ CLI tool for managing documentation sources (`crawl_docs.py`)
- ✅ Crawled 314 pages from 3 Adobe Spectrum sources

**Sources Indexed:**
- Spectrum Web Components (123 pages)
- React Spectrum (89 pages)
- Spectrum Design System (104 pages)

**Files Created:**
- `src/ingestion/web_crawler.py` (379 lines)
- `src/ingestion/crawl_docs.py` (246 lines)
- `config/crawler_config.yaml`
- `src/ingestion/CRAWLER.md` (documentation)

### ✅ Phase 2b: Document Processing (Completed)
- ✅ Built markdown parser with YAML frontmatter support
- ✅ Implemented intelligent chunking strategy (200-1500 chars)
- ✅ Code block preservation with `[code]...[/code]` detection
- ✅ Section-based chunking using markdown headings
- ✅ Metadata preservation with each chunk
- ✅ Created 2,161 chunks from 314 documents

**Chunking Strategy:**
- Preserves code blocks intact (never splits mid-code)
- Groups content by markdown headings
- Maintains component metadata (title, URL, domain, heading)
- Respects size limits while keeping logical units together

**Files Created:**
- `src/ingestion/document_processor.py` (348 lines)

### ✅ Phase 3: Indexing & Retrieval (Completed)
- ✅ Generated embeddings using sentence-transformers
- ✅ Indexed 2,161 chunks in Chroma vector database
- ✅ Stored rich metadata for filtering (domain, title, heading, chunk_type)
- ✅ Created persistent vector store at `./data/chroma_db` (41MB)
- ✅ Implemented batch processing (50 docs/batch)
- ✅ Semantic similarity search with top-k retrieval

**Vector Database Stats:**
- 2,147 documents indexed
- 384-dimensional embeddings (all-MiniLM-L6-v2)
- Collection name: `design_system_docs`
- Indexing time: ~43 seconds

**Files Created:**
- `src/ingestion/document_indexer.py` (201 lines)

### ✅ Phase 4: Query Pipeline & Generation (Completed)
- ✅ Built end-to-end RAG pipeline with Haystack
- ✅ Integrated Claude Sonnet 4.5 for response generation
- ✅ Semantic search with Chroma embedding retrieval (top-5)
- ✅ Chat-based prompt builder with system/user messages
- ✅ Context-aware responses with source citations
- ✅ Tested with design system queries (buttons, colors, components)

**RAG Pipeline Components:**
1. Query embedding (sentence-transformers)
2. Vector similarity search (Chroma)
3. Prompt building with retrieved context
4. Response generation (Claude)

**Example Query Results:**
- Query: "How do I use a button component in React Spectrum?"
- Retrieved: 5 relevant documents
- Response: Comprehensive answer with code examples, installation steps, and source citations
- Relevance score: 0.599 (top match)

**Files Created:**
- `src/query/rag_pipeline.py` (255 lines)

### ✅ Phase 5: CLI & API (Completed)
- ✅ Create user-friendly CLI query tool with Rich formatting
- ✅ Implement REST API with FastAPI
- ✅ Add /query, /health, /stats, /refresh endpoints
- ✅ Support domain filtering and top-k adjustment
- ✅ Add comprehensive test coverage (71 tests)
- [ ] Build evaluation framework with test queries
- [ ] Add logging and monitoring
- [ ] Implement caching for frequent queries
- [ ] Add rate limiting
- [ ] Deploy with Docker compose

**Files Created:**
- `query.py` (269 lines) - Rich CLI interface
- `src/api/server.py` (424 lines) - FastAPI REST API
- `tests/test_rag_pipeline.py` (469 lines, 21 tests)
- `tests/test_document_indexer.py` (473 lines, 25 tests)
- `tests/test_document_processor.py` (updated with 20 tests)

### ✅ Phase 6: Modular Architecture (Completed)

**Objective:** Refactor monolithic code into extensible, adapter-based architecture to support multiple ingestion sources, embedding providers, LLMs, and query interfaces.

#### Phase 6.1: Base Abstractions (Completed)
Created abstract base classes for all major components to enable swappable implementations:

**1. Ingestion Adapter Interface** (`src/ingestion/adapters/base.py` - 267 lines)
- Abstract base class `IngestionAdapter` for all document sources
- Methods: `connect()`, `list_documents()`, `fetch_document()`, `fetch_all()`, `get_updates_since()`
- Dataclasses: `Document`, `DocumentMetadata`
- Supports: Web crawlers, file systems, CMS platforms, databases
- Context manager support for resource cleanup

**2. Embedding Provider Interface** (`src/embedding/providers/base.py` - 294 lines)
- Abstract base class `EmbeddingProvider` for embedding models
- Methods: `embed_text()`, `embed_batch()`, `embed_documents()`
- Dataclasses: `EmbeddingConfig`, `EmbeddingResult`
- Supports: Sentence Transformers, OpenAI, Cohere, custom models
- Configurable batching and normalization

**3. LLM Provider Interface** (`src/generation/providers/base.py` - 373 lines)
- Abstract base class `LLMProvider` for language models
- Methods: `generate()`, `chat()`, `generate_with_context()`, `generate_stream()`
- Dataclasses: `LLMConfig`, `ChatMessage`, `GenerationResult`
- Supports: Anthropic, OpenAI, Cohere, local models
- Async streaming support

**4. Chunking Strategy Interface** (`src/processing/chunkers/base.py` - 294 lines)
- Abstract base class `ChunkerStrategy` for text chunking
- Methods: `chunk_text()`, `chunk_documents()`, `validate_chunk()`, `merge_small_chunks()`
- Dataclasses: `ChunkingConfig`, `Chunk`, `ChunkType` enum
- Supports: Fixed-size, semantic, markdown-aware strategies
- Code block and table preservation

**5. Retrieval Strategy Interface** (`src/retrieval/strategies/base.py` - 391 lines)
- Abstract base class `RetrievalStrategy` for document retrieval
- Methods: `retrieve()`, `compute_similarity()`, `rerank_documents()`, `promote_diversity()`
- Dataclasses: `RetrievalConfig`, `RetrievedDocument`, `RetrievalResult`
- Supports: Vector similarity, hybrid search, BM25, reranking
- Diversity promotion and score-based filtering

**6. Query Interface Protocol** (`src/query/interfaces/base.py` - 299 lines)
- Abstract base class `QueryInterface` for query endpoints
- Methods: `process_query()`, `format_response()`, `validate_request()`, `process_query_stream()`
- Dataclasses: `QueryRequest`, `QueryResponse`, `SourceDocument`
- Supports: CLI, REST API, Custom GPT, OpenAI-compatible API
- Error handling and streaming support

**Total Base Abstraction Code:** 1,918 lines across 6 modules

#### Phase 6.2: Refactoring to Modular Architecture (In Progress)
Using Test-Driven Development to refactor existing code into new modular structure:

**Completed Refactorings:**
1. ✅ **MarkdownChunker** (`src/processing/chunkers/markdown.py` - 227 lines)
   - Implements `ChunkerStrategy` interface
   - Extracted chunking logic from `DocumentProcessor`
   - All 20 document processor tests pass
   - Maintains backward compatibility with `DocumentChunk` format

2. ✅ **SentenceTransformersProvider** (`src/embedding/providers/sentence_transformers.py` - 224 lines)
   - Implements `EmbeddingProvider` interface
   - Uses sentence-transformers library directly
   - Context manager support for resource cleanup
   - 29 comprehensive unit tests (all passing)
   - Handles edge cases: empty text, batch processing, dimension mismatch

3. ✅ **DocumentIndexer Refactoring** (updated 226 → 267 lines)
   - Supports both legacy Haystack mode and new EmbeddingProvider mode
   - Backward compatible: all 25 original tests pass
   - New provider mode: 11 additional integration tests pass
   - Dual-mode implementation: `_index_with_haystack()` + `_index_with_provider()`
   - Optional `embedding_provider` parameter enables modular architecture

**Pending Refactorings:**
4. ⏳ **AnthropicProvider** (LLM provider)
   - Extract Claude integration from `rag_pipeline.py`
   - Implement `LLMProvider` interface
   - Maintain all 21 RAG pipeline tests passing

5. ⏳ **VectorSimilarityRetriever** (retrieval strategy)
   - Extract Chroma retrieval logic
   - Implement `RetrievalStrategy` interface

**Test Coverage:** 111 total tests (all passing) ⬆️ +40 tests
- 20 tests: Document processing
- 25 tests: Document indexing (legacy mode)
- 11 tests: Document indexing (provider mode)
- 29 tests: SentenceTransformersProvider
- 21 tests: RAG pipeline
- 5 tests: Setup verification

#### Future Extensibility (Planned)

With the modular architecture in place, the system will support:

**Ingestion Adapters:**
- ✅ Web crawler (existing)
- 📋 File system reader
- 📋 CMS integration (Contentful, Strapi)
- 📋 Database reader (Postgres, MongoDB)

**Embedding Providers:**
- ✅ Sentence Transformers (modular implementation complete)
- ✅ OpenAI embeddings (modular implementation complete)
- 📋 Cohere embeddings
- 📋 HuggingFace Inference API

**LLM Providers:**
- ✅ Anthropic Claude (modular implementation complete)
- 📋 OpenAI GPT-4
- 📋 OpenAI GPT-3.5
- 📋 Local models via Ollama

**Query Interfaces:**
- ✅ CLI (existing)
- ✅ REST API (existing)
- 📋 Custom GPT Actions API
- 📋 OpenAI-compatible API

**Retrieval Strategies:**
- ✅ Vector similarity (existing)
- 📋 Hybrid search (vector + keyword)
- 📋 BM25 keyword search
- 📋 Reranking with cross-encoders

## Phase 7: Codebase Ingestion - Entity-Level Indexing

**Objective:** Enable querying actual source code implementations alongside documentation by integrating code parsers with the ingestion pipeline.

### Current State (Phase 7.1 - Foundation Complete ✅)

**Code Parsing Infrastructure:**
- ✅ `PythonParser` - AST-based parsing for reliable entity extraction (204 lines)
- ✅ `JavaScriptParser` - Regex-based parsing for JS/TS (277 lines)
- ✅ `CodeParser` base interface with `CodeEntity` dataclass (145 lines)
- ✅ `EntityType` enum: FUNCTION, CLASS, METHOD, VARIABLE, CONSTANT, etc.
- ✅ 11 tests covering both parsers (100% passing)

**Codebase Ingestion:**
- ✅ `CodebaseAdapter` - Directory traversal with language detection (430 lines)
- ✅ `CodeChunker` - Language-aware code chunking (450 lines)
- ✅ Support for 11+ programming languages
- ✅ File filtering, incremental updates, metadata extraction
- ✅ 13 tests covering adapter and chunker (100% passing)

**Current Limitation:**
- Parsers and adapter are **not integrated**
- Adapter ingests whole files, not individual entities
- No entity-level metadata (signatures, parameters, docstrings)
- CodeChunker uses regex, doesn't leverage parsed entity boundaries

### Phase 7.2: Entity-Level Integration (Complete ✅)

**Goal:** Enable queries like:
- *"Show me the Button component implementation"* → Returns Button class/function with signature
- *"What parameters does authenticate() accept?"* → Returns function signature + docstring
- *"How do I customize Button colors?"* → Returns docs + color-related props from source

**Implementation Plan:**

**1. Create CodeParserRegistry** (`src/ingestion/parsers/registry.py`)
```python
class CodeParserRegistry:
    """Maps programming languages to appropriate parsers."""
    - register_parser(language, parser_class)
    - get_parser(language) -> CodeParser
    - supports_language(language) -> bool
    - Lazy-load parsers for performance
    - Built-in support for Python, JavaScript/TypeScript
```

**2. Create CodeEntityFormatter** (`src/ingestion/formatters/code_entity_formatter.py`)
```python
class CodeEntityFormatter:
    """Converts CodeEntity objects to Document objects with rich metadata."""
    - format_entity(entity: CodeEntity, file_path: str) -> Document
    - Metadata includes:
        - entity_type, entity_name, signature
        - parameters, return_type, decorators
        - file_path, programming_language
        - parent_entity (for methods)
        - docstring
    - Content includes full entity code
```

**3. Enhance CodebaseAdapter**
```python
# Add new method to CodebaseAdapter
def parse_with_entities(
    self,
    file_path: str,
    parser_registry: CodeParserRegistry
) -> List[Document]:
    """
    Parse a code file into individual entity documents.
    Returns one Document per function/class/method.
    """
```

**4. Integration Tests** (`tests/test_code_entity_integration.py`)
- End-to-end test: directory → entities → indexed → queried
- Verify metadata preservation through pipeline
- Test with Python and JavaScript files
- Validate entity-level retrieval accuracy

**Files Created:**
- ✅ `src/ingestion/parsers/registry.py` (187 lines) - Language-to-parser mapping with lazy loading
- ✅ `src/ingestion/formatters/` (new directory)
- ✅ `src/ingestion/formatters/__init__.py` - Module exports
- ✅ `src/ingestion/formatters/code_entity_formatter.py` (251 lines) - Entity-to-Document conversion
- ✅ `tests/test_code_entity_integration.py` (379 lines, 11 tests)

**Files Modified:**
- ✅ `src/ingestion/adapters/codebase.py` - Added `parse_with_entities()` and `fetch_all_entities()` methods (+144 lines)
- ✅ `src/ingestion/parsers/__init__.py` - Export CodeParserRegistry

**Outcome:**
- ✅ Entity-level code indexing: 1 document per function/class
- ✅ Rich metadata for precise retrieval (signatures, parameters, return types, docstrings)
- ✅ Queries return exact implementations with context
- ✅ Maintains backward compatibility (whole-file mode still available)
- ✅ 11 integration tests passing (100%)

**Total Code:** ~820 lines of new code, 11 tests

### Phase 7.3: Advanced Code Features (Future 📋)

**Planned Enhancements:**
- 📋 Improved JavaScript/TypeScript parser (replace regex with tree-sitter)
- 📋 Additional language parsers (Java, Go, Rust)
- 📋 Cross-file entity references (imports, inheritance)
- 📋 Entity relationship tracking (which functions call which)
- 📋 Code context enrichment (include imports, type definitions)
- 📋 Multi-file context (class definition + method implementations)

## Future Features & Enhancements

### Option A: Additional Embedding Providers
- 📋 Cohere embeddings (multilingual support)
- 📋 HuggingFace Inference API embeddings
- 📋 Azure OpenAI embeddings
- 📋 Vertex AI embeddings
- 📋 Local embedding models via Ollama

### Option B: Additional LLM Providers
- 📋 OpenAI GPT-4 provider
- 📋 OpenAI GPT-3.5 provider
- 📋 Local LLMs via Ollama
- 📋 Azure OpenAI provider
- 📋 Cohere Command models

### Option C: Advanced Retrieval Features
- 📋 Hybrid search (semantic + keyword/BM25)
- 📋 Reranking with cross-encoders
- 📋 Query expansion
- 📋 Embedding caching layer
- 📋 Multi-vector retrieval
- 📋 Metadata filtering improvements

### Option D: Production Enhancements
- 📋 Authentication & authorization (API keys, OAuth)
- 📋 Rate limiting middleware
- 📋 Monitoring & observability (metrics, traces)
- 📋 Caching layer (Redis)
- 📋 Async/streaming support
- 📋 Multi-tenancy support
- 📋 Health checks & circuit breakers

### Option E: Developer Experience
- 📋 CLI improvements (interactive config, provider selection)
- 📋 Docker Compose for full stack
- 📋 Configuration UI
- 📋 Evaluation framework (accuracy metrics)
- 📋 A/B testing framework
- 📋 Migration scripts (legacy → modular)

### Option F: Query Interfaces
- 📋 Custom GPT Actions API integration
- 📋 OpenAI-compatible API endpoint
- 📋 GraphQL API
- 📋 WebSocket support for streaming
- 📋 Slack/Discord bot integration

## Implementation Summary

**Total Code Written:** ~8,620+ lines across 29+ modules (⬆️ +820 lines in Phase 7.2)
**Documentation Indexed:** 314 pages → 2,147 searchable chunks
**Test Coverage:** 194 tests (100% passing) ⬆️ +35 tests since Phase 7 start (159 → 194)
**Technologies Used:** Haystack, Chroma, Anthropic Claude, OpenAI, Playwright, sentence-transformers, FastAPI, Rich
**Architecture:** Modular adapter-based design for extensibility with entity-level code ingestion

**Recent Updates (Phase 6.2 - Multi-Provider Architecture):**

**Embedding Providers:**
- ✅ Added `SentenceTransformersProvider` with full EmbeddingProvider interface (224 lines, 29 tests)
- ✅ Added `OpenAIEmbeddingProvider` with API integration (340 lines, 26 tests)
- ✅ Created `EmbeddingProviderFactory` for easy provider instantiation (260 lines)
- ✅ Refactored `DocumentIndexer` to support dual-mode operation (backward compatible)
- ✅ Created comprehensive test suite for provider mode (37 integration tests)
- ✅ Added comparison script demonstrating both providers
- ✅ Wrote detailed documentation guide (EMBEDDING_PROVIDERS.md)

**LLM Providers:**
- ✅ Added `AnthropicProvider` with full LLMProvider interface (400 lines, 22 tests)
- ✅ Full Claude API integration (3.5 Sonnet, Opus, Haiku support)
- ✅ Streaming support with `generate_stream()`
- ✅ RAG-optimized with `generate_with_context()` helper
- ✅ Context manager support and proper error handling

**Retrieval Strategies:**
- ✅ Added `ChromaRetriever` with full RetrievalStrategy interface (450 lines)
- ✅ Vector similarity search with embedding provider integration
- ✅ Metadata filtering and score thresholds
- ✅ Diversity promotion (MMR-like approach)
- ✅ Reranking support (placeholder for future cross-encoder)
- ✅ Direct Chroma client integration for efficient searches

**Overall Progress:**
- ✅ Maintained 100% test pass rate throughout (159 tests passing)
- ✅ Demonstrated real user value: easy provider switching for embeddings, LLMs, and retrieval
- ✅ Backward compatible: all existing code continues to work
- ✅ Production-ready implementations with comprehensive error handling
- ✅ **5 of 6 planned refactorings complete** (MarkdownChunker, SentenceTransformersProvider, OpenAIEmbeddingProvider, AnthropicProvider, ChromaRetriever)
