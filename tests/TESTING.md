# Testing and Demo Guide

This guide explains how to verify your RAG pipeline setup and run demos.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your ANTHROPIC_API_KEY
# You can get one from: https://console.anthropic.com/
```

### 3. Run Setup Verification

```bash
python tests/test_setup.py
```

This will test:
- ✓ Package imports (Haystack, ChromaDB, Anthropic)
- ✓ Environment configuration (API keys)
- ✓ ChromaDB connection
- ✓ Basic Haystack pipeline
- ✓ Anthropic Claude integration

### 4. Run Simple RAG Demo

```bash
python demo_simple_rag.py
```

This demonstrates:
- Document embedding with sentence-transformers
- Semantic search and retrieval
- Response generation with Claude
- End-to-end RAG pipeline with sample design system docs

## What the Demo Does

The demo creates a mini design system documentation search:

1. **Sample Documentation**: Includes 4 sample docs:
   - Button Component (Actions)
   - Input Component (Forms)
   - Card Component (Layout)
   - Design Tokens - Spacing

2. **Embeds Documents**: Uses sentence-transformers to create vector embeddings

3. **Runs Sample Queries**:
   - "How do I use the Button component with different variants?"
   - "What are the spacing tokens available?"
   - "Show me how to create a form input with error handling"

4. **Retrieves & Generates**:
   - Finds relevant docs using semantic search
   - Generates natural language answers using Claude
   - Shows which components were retrieved

## Expected Output

```
Simple RAG Pipeline Demo - Design System Documentation
======================================================================

1. Setting up document store and embedder...
   Created 4 sample documents

2. Generating embeddings...
   ✓ Embedded and stored 4 documents

3. Setting up retrieval pipeline...
   ✓ Retrieval components ready

4. Initializing Claude for response generation...
   ✓ Claude ready

Running Demo Queries
======================================================================

Query 1: How do I use the Button component with different variants?
Retrieved 2 relevant documents:
  1. Button (Actions)
  2. Input (Forms)

Response:
[Claude's answer with code examples from the docs]
...
```

## Troubleshooting

### "ANTHROPIC_API_KEY not found"
- Make sure you've created `.env` file from `.env.example`
- Add your API key: `ANTHROPIC_API_KEY=sk-ant-...`

### "ImportError: No module named 'haystack'"
- Run: `pip install -r requirements.txt`

### "Model download is slow"
- First run downloads the sentence-transformers model (~80MB)
- Subsequent runs will use the cached model

### ChromaDB connection issues
- For the demo, we use in-memory ChromaDB (no Docker needed)
- For production, start Docker: `docker-compose up -d`

## Next Steps

After verifying the setup works:

1. **Add Real Documentation**: Place your design system docs in `data/raw/`
2. **Build Ingestion Pipeline**: Process and chunk your actual docs
3. **Use Production Vector Store**: Switch to Chroma with Docker
4. **Add Metadata Filtering**: Filter by component, category, version
5. **Build API**: Create REST endpoints for queries

## Running with Docker

To test with the full Docker stack:

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# The setup verification won't work inside Docker yet
# (API endpoints not implemented)
```

## Performance Notes

- **Embedding Model**: Using `all-MiniLM-L6-v2` (fast, good quality)
- **First Query**: ~2-3 seconds (includes model loading)
- **Subsequent Queries**: ~500ms-1s (model cached)
- **Claude API**: ~1-2 seconds per response

## Demo Limitations

This is a minimal demo. Production system should add:
- Persistent vector storage (Chroma with Docker)
- Batch processing for large doc sets
- Metadata filtering and advanced search
- Caching for common queries
- Error handling and retries
- Monitoring and logging
