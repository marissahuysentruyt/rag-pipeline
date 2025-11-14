# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) pipeline for design system documentation search, built with:
- **Haystack** (`haystack-ai`) - Modern RAG framework
- **Anthropic Claude** (`anthropic-haystack`) - LLM integration for response generation
- **Docker** - Containerized deployment with vector database

## Architecture

### Core Components

**Pipeline-based Architecture (Haystack)**
- `src/ingestion/` - Document loaders for Markdown, MDX, HTML, JSON (design tokens)
- `src/processing/` - Component-aware chunking with metadata preservation
- `src/embedding/` - Embedding generation and vector storage
- `src/retrieval/` - Hybrid search (semantic + keyword) with metadata filtering
- `src/generation/` - Claude-powered response generation
- `src/api/` - Query interface (REST API or CLI)

**Data Flow**
1. Documentation ingestion → 2. Chunking with metadata → 3. Embedding generation → 4. Vector storage → 5. Query → 6. Retrieval → 7. Claude response generation

**Vector Database Options**
- Development: FAISS (local, no separate service)
- Production: Chroma (Docker-compatible, persistent)
- Alternative: Qdrant, Weaviate, or Pinecone

### Design System Specifics

The pipeline is optimized for design system documentation:
- Component-centric chunking (keep props, examples, guidelines together)
- Preserves hierarchical structure (category → component → props)
- Code-aware processing (syntax highlighting, framework detection)
- Metadata indexing (component type, category, version, tags)

## Development Commands

### Setup

```bash
# Local development
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your ANTHROPIC_API_KEY

# Docker deployment
docker-compose up -d
```

### Running the Pipeline

```bash
# Run API server (local)
python -m src.api.main

# Run with Docker
docker-compose up -d

# View logs
docker-compose logs -f rag-app
```

### Testing & Quality

```bash
# Run tests
pytest tests/

# Format code
black src/

# Lint code
ruff check src/
```

### Docker Commands

```bash
# Build and start services
docker-compose up -d --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Rebuild specific service
docker-compose up -d --build rag-app
```

## Configuration

Environment variables in `.env`:
- `ANTHROPIC_API_KEY` - Required for Claude integration
- `OPENAI_API_KEY` - Optional, for OpenAI embeddings
- `CHUNK_SIZE` - Document chunk size (default: 512)
- `CHUNK_OVERLAP` - Chunk overlap size (default: 50)
- `EMBEDDING_MODEL` - Embedding model name
- `LLM_MODEL` - Claude model (e.g., claude-3-5-sonnet-20241022)

See `.env.example` for full configuration options.
