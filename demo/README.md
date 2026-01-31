# RAG Pipeline Demo

Interactive demos showcasing each phase of the RAG (Retrieval-Augmented Generation) pipeline.

## Quick Start

### Run Full Pipeline

```bash
# Complete demo (requires ANTHROPIC_API_KEY in .env)
python demo/run_full_pipeline.py

# Skip AI generation phase (no API key needed)
python demo/run_full_pipeline.py --skip-generation
```

### Run Individual Phases

Each phase can be run independently:

```bash
# Phase 1: Document Ingestion
python demo/phase_1_ingest.py

# Phase 2: Semantic Chunking
python demo/phase_2_chunk.py

# Phase 3: Embedding Generation
python demo/phase_3_embed.py

# Phase 4: Vector Indexing
python demo/phase_4_index.py

# Phase 5: Semantic Retrieval
python demo/phase_5_retrieve.py

# Phase 6: AI Response Generation (requires API key)
python demo/phase_6_generate.py
```

## What Each Phase Demonstrates

| Phase | Name | What It Shows |
|-------|------|---------------|
| 1 | **Ingestion** | Loading markdown files, parsing YAML frontmatter, extracting metadata |
| 2 | **Chunking** | Splitting documents into semantic chunks while preserving code blocks and structure |
| 3 | **Embedding** | Converting text to vectors, showing dimensions and semantic similarity |
| 4 | **Indexing** | Storing embeddings in Chroma vector database |
| 5 | **Retrieval** | Semantic search with sample queries, ranked results |
| 6 | **Generation** | Claude generating answers using retrieved context |

## Sample Data: Golden Design System

The demos use a fictional design system called "Golden" with 4 components:

**Documentation** (`data/golden/docs/`):
- `button.md` - Button component with variants, props, guidelines
- `card.md` - Card container component
- `modal.md` - Modal dialog with focus management
- `input.md` - Form input with validation

**Implementation** (`data/golden/components/`):
- `Button.tsx`, `Card.tsx`, `Modal.tsx`, `Input.tsx`

## Requirements

```bash
# Install dependencies (from project root)
pip install -r requirements.txt

# For Phase 6, set API key in .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
```

## Output

Each phase displays:
- **Phase header** - Colored banner with description
- **Summary table** - Key statistics (documents processed, chunks created, etc.)
- **Sample output** - Preview of actual data (metadata, chunk content, embeddings)

The output uses `rich` for terminal formatting with colors, tables, and panels.
