# Migration Plan: Remove Haystack Dependency

## Overview

This document outlines a plan to remove the Haystack framework dependency while maintaining all current functionality. The goal is to have a fully custom RAG pipeline using only the modular architecture we've built.

---

## Current Haystack Dependencies

### Where Haystack is Currently Used

1. **Document Indexing** (`src/ingestion/document_indexer.py`)
   - Uses `SentenceTransformersDocumentEmbedder` from Haystack
   - Legacy mode falls back to Haystack's embedding component
   - Dual-mode: Haystack embedder OR our EmbeddingProvider interface

2. **RAG Pipeline** (`src/query/rag_pipeline.py`)
   - Uses `Pipeline` class from Haystack for orchestration
   - Uses `ChromaQueryTextRetriever` for vector search
   - Uses `PromptBuilder` for prompt construction
   - Uses `ChatPromptBuilder` for chat-style prompts

3. **Dependencies in requirements.txt**
   - `haystack-ai>=2.0.0`
   - `chroma-haystack>=0.20.0`
   - `anthropic-haystack>=0.0.1`

---

## What We've Already Built (No Changes Needed)

These components are **Haystack-independent** and ready to use:

### ✅ Ingestion Layer
- `CodebaseAdapter` - Source code ingestion
- `CodeParserRegistry` - Language-to-parser mapping
- `PythonParser`, `JavaScriptParser` - Entity extraction
- `CodeEntityFormatter` - Entity-to-Document conversion
- `WebCrawler` - Documentation crawling
- **Status:** 100% custom, no Haystack

### ✅ Processing Layer
- `MarkdownChunker` - Markdown-aware chunking
- `CodeChunker` - Language-aware code chunking
- `ChunkerStrategy` base interface
- **Status:** 100% custom, no Haystack

### ✅ Embedding Providers
- `SentenceTransformersProvider` - Local embeddings
- `OpenAIEmbeddingProvider` - OpenAI API embeddings
- `EmbeddingProviderFactory` - Factory pattern
- `EmbeddingProvider` base interface
- **Status:** 100% custom, no Haystack

### ✅ LLM Providers
- `AnthropicProvider` - Claude integration
- `LLMProvider` base interface
- **Status:** 100% custom, no Haystack

### ✅ Retrieval Strategies
- `ChromaRetriever` - Direct Chroma client usage
- `RetrievalStrategy` base interface
- **Status:** 100% custom, no Haystack

### ✅ Query Interfaces
- CLI (`query.py`) - Rich terminal interface
- REST API (`src/api/server.py`) - FastAPI endpoints
- **Status:** Uses RAG pipeline, but interface code is Haystack-independent

---

## What Needs to Change

### 1. Document Indexer (Moderate Refactor)

**Current State:**
- Dual-mode: Haystack embedder OR EmbeddingProvider
- Haystack mode uses `SentenceTransformersDocumentEmbedder`

**Required Changes:**
- Remove Haystack mode entirely
- Make `embedding_provider` **required** (not optional)
- Update all calling code to pass EmbeddingProvider

**Affected Files:**
- `src/ingestion/document_indexer.py` - Remove `_index_with_haystack()`, simplify
- Tests: `tests/test_document_indexer.py` - Update to always use providers

**Estimated Effort:** 2-3 hours (mostly test updates)

---

### 2. RAG Pipeline Orchestration (Major Refactor)

**Current State:**
```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.components.retrievers import ChromaQueryTextRetriever

pipeline = Pipeline()
pipeline.add_component("retriever", ChromaQueryTextRetriever(...))
pipeline.add_component("prompt_builder", PromptBuilder(...))
pipeline.add_component("llm", AnthropicChatGenerator(...))
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm.messages")
```

**Required Changes:**

#### A. Create Custom Pipeline Orchestrator

```python
# src/query/pipeline_orchestrator.py

class RAGPipelineOrchestrator:
    """
    Custom RAG pipeline orchestrator (replaces Haystack Pipeline).

    Coordinates:
    1. Query embedding
    2. Document retrieval
    3. Prompt building
    4. LLM generation
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        retrieval_strategy: RetrievalStrategy,
        llm_provider: LLMProvider,
        prompt_template: Optional[str] = None
    ):
        self.embedding_provider = embedding_provider
        self.retrieval_strategy = retrieval_strategy
        self.llm_provider = llm_provider
        self.prompt_template = prompt_template or self._default_prompt()

    def run(self, query: str, filters: Optional[Dict] = None, top_k: int = 5):
        """
        Execute RAG pipeline.

        Steps:
        1. Embed query
        2. Retrieve documents
        3. Build prompt with context
        4. Generate response
        """
        # 1. Embed query
        query_embedding = self.embedding_provider.embed_text(query)

        # 2. Retrieve documents
        retrieval_result = self.retrieval_strategy.retrieve(
            query=query,
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k
        )

        # 3. Build context from retrieved documents
        context = self._build_context(retrieval_result.documents)

        # 4. Build prompt
        prompt = self.prompt_template.format(
            query=query,
            context=context
        )

        # 5. Generate response
        response = self.llm_provider.generate(prompt)

        return {
            "answer": response.text,
            "sources": retrieval_result.documents,
            "metadata": {
                "query": query,
                "num_documents": len(retrieval_result.documents),
                "model": self.llm_provider.model_name
            }
        }

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """Format retrieved documents as context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Title: {doc.metadata.get('title', 'Unknown')}\n"
                f"Content: {doc.content}\n"
            )
        return "\n\n".join(context_parts)

    def _default_prompt(self) -> str:
        """Default RAG prompt template."""
        return """You are a helpful assistant answering questions about design system documentation.

Context from documentation:
{context}

User question: {query}

Please provide a helpful answer based on the context above. Include code examples where relevant."""
```

#### B. Simplify RAG Pipeline

**New approach:**
```python
# src/query/rag_pipeline.py (simplified)

class RAGPipeline:
    """
    Simplified RAG pipeline without Haystack.
    """

    def __init__(
        self,
        collection_name: str,
        chroma_persist_path: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        retrieval_strategy: Optional[RetrievalStrategy] = None
    ):
        # Create defaults if not provided
        self.embedding_provider = embedding_provider or create_embedding_provider(
            "sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.llm_provider = llm_provider or AnthropicProvider()

        self.retrieval_strategy = retrieval_strategy or ChromaRetriever(
            collection_name=collection_name,
            persist_directory=chroma_persist_path,
            embedding_provider=self.embedding_provider
        )

        # Create orchestrator
        self.orchestrator = RAGPipelineOrchestrator(
            embedding_provider=self.embedding_provider,
            retrieval_strategy=self.retrieval_strategy,
            llm_provider=self.llm_provider
        )

    def query(
        self,
        question: str,
        domain: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Execute RAG query."""
        filters = {"domain": domain} if domain else None
        return self.orchestrator.run(question, filters=filters, top_k=top_k)
```

**Affected Files:**
- `src/query/rag_pipeline.py` - Complete rewrite (simpler!)
- `src/query/pipeline_orchestrator.py` - New file
- Tests: `tests/test_rag_pipeline.py` - Update for new structure

**Estimated Effort:** 4-6 hours

---

### 3. Remove Haystack Dependencies

**Files to Update:**

#### requirements.txt
Remove:
```
haystack-ai>=2.0.0
chroma-haystack>=0.20.0
anthropic-haystack>=0.0.1
```

Keep:
```
chromadb>=0.4.22
sentence-transformers>=2.2.2
anthropic>=0.18.0
openai>=1.12.0
# ... all other dependencies
```

#### All Import Statements
Replace:
```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers import ChromaQueryTextRetriever
```

With:
```python
from src.query.pipeline_orchestrator import RAGPipelineOrchestrator
from src.retrieval.strategies import ChromaRetriever
from src.generation.providers import AnthropicProvider
```

**Affected Files:**
- `src/query/rag_pipeline.py`
- `src/ingestion/document_indexer.py`
- `src/api/server.py` (minimal - just imports)
- `query.py` (minimal - just imports)
- All test files that import Haystack components

**Estimated Effort:** 1-2 hours (mostly find/replace)

---

## Migration Strategy

### Phase 1: Add New Components (No Breaking Changes)

**Goal:** Build replacements alongside existing Haystack code

**Steps:**
1. Create `src/query/pipeline_orchestrator.py` (new)
2. Add tests for `RAGPipelineOrchestrator`
3. Create `src/query/rag_pipeline_v2.py` (new, Haystack-free version)
4. Add tests for new pipeline
5. Verify both old and new pipelines work side-by-side

**Result:** Can switch between Haystack and custom implementations

**Estimated Time:** 1 day

---

### Phase 2: Update DocumentIndexer (Breaking Change)

**Goal:** Remove Haystack dependency from indexing

**Steps:**
1. Remove `_index_with_haystack()` method
2. Make `embedding_provider` required parameter
3. Update all calling code to pass provider
4. Update tests to use providers
5. Verify all 194 tests still pass

**Result:** Indexing is 100% custom

**Estimated Time:** 4 hours

---

### Phase 3: Switch RAG Pipeline (Breaking Change)

**Goal:** Replace Haystack pipeline with custom orchestrator

**Steps:**
1. Replace `src/query/rag_pipeline.py` with v2 implementation
2. Update `src/api/server.py` to use new pipeline
3. Update `query.py` CLI to use new pipeline
4. Update all RAG pipeline tests
5. Verify end-to-end query flow works

**Result:** RAG queries work without Haystack

**Estimated Time:** 6 hours

---

### Phase 4: Cleanup (No Breaking Changes)

**Goal:** Remove all Haystack references

**Steps:**
1. Remove Haystack from `requirements.txt`
2. Remove unused imports
3. Delete any Haystack-related helper code
4. Update documentation (README, CLAUDE.md)
5. Run full test suite
6. Test Docker build without Haystack

**Result:** Zero Haystack dependencies

**Estimated Time:** 2 hours

---

## Benefits of Removing Haystack

### Pros
1. **Full Control** - No framework abstractions, direct control over every step
2. **Smaller Dependencies** - Remove haystack-ai, chroma-haystack, anthropic-haystack
3. **Simpler Code** - Pipeline orchestration is ~150 lines vs Haystack's complexity
4. **Easier Debugging** - No framework magic, straightforward execution flow
5. **Better Documentation** - Our code is easier to understand than Haystack docs
6. **Flexibility** - Easy to customize any step without fighting the framework

### Cons
1. **Loss of Haystack Ecosystem** - Can't easily add Haystack components
2. **Manual Pipeline Management** - Have to build orchestration ourselves
3. **More Code to Maintain** - Responsible for all orchestration logic
4. **No Haystack Updates** - Won't benefit from Haystack improvements

---

## Code Comparison

### Current (With Haystack)

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers import ChromaQueryTextRetriever
from anthropic_haystack import AnthropicChatGenerator

# Build pipeline
pipeline = Pipeline()
pipeline.add_component("retriever", ChromaQueryTextRetriever(...))
pipeline.add_component("prompt_builder", ChatPromptBuilder(...))
pipeline.add_component("llm", AnthropicChatGenerator(...))
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.messages")

# Run
result = pipeline.run({
    "retriever": {"query": question},
    "prompt_builder": {"question": question}
})
```

### After Migration (Without Haystack)

```python
from src.query.pipeline_orchestrator import RAGPipelineOrchestrator
from src.retrieval.strategies import ChromaRetriever
from src.generation.providers import AnthropicProvider
from src.embedding.factory import create_embedding_provider

# Create components
embedding_provider = create_embedding_provider("sentence-transformers")
retriever = ChromaRetriever(collection_name="docs", embedding_provider=embedding_provider)
llm = AnthropicProvider()

# Create orchestrator
orchestrator = RAGPipelineOrchestrator(
    embedding_provider=embedding_provider,
    retrieval_strategy=retriever,
    llm_provider=llm
)

# Run (simpler!)
result = orchestrator.run(question, top_k=5)
```

**Less code, more clarity!**

---

## Testing Strategy

### Test Coverage During Migration

1. **Keep all existing tests passing** during Phase 1 & 2
2. **Parallel testing** - Test both Haystack and custom pipelines
3. **Integration tests** - End-to-end queries with actual Chroma + Claude
4. **Performance testing** - Ensure custom pipeline is as fast as Haystack

### Test Files to Update

- `tests/test_document_indexer.py` - Remove Haystack mode tests
- `tests/test_rag_pipeline.py` - Complete rewrite for new orchestrator
- `tests/test_query_api.py` - Update for new pipeline interface
- `tests/test_setup.py` - Remove Haystack dependency checks

---

## Risk Assessment

### Low Risk
- ✅ Ingestion/processing/embedding code already custom
- ✅ Most tests don't depend on Haystack
- ✅ API/CLI interfaces are abstracted from pipeline details

### Medium Risk
- ⚠️ RAG pipeline is core functionality - bugs affect all queries
- ⚠️ Prompt building needs careful testing (Haystack's PromptBuilder is well-tested)
- ⚠️ Document formatting for context needs to match current behavior

### Mitigation
- Run old and new pipelines in parallel during Phase 1
- Compare outputs for identical queries
- Keep Haystack version in git history for rollback

---

## Estimated Total Effort

**Total Time:** 2-3 days of focused work

**Breakdown:**
- Phase 1 (New components): 8 hours
- Phase 2 (IndexerUpdate): 4 hours
- Phase 3 (Pipeline switch): 6 hours
- Phase 4 (Cleanup): 2 hours
- **Total:** ~20 hours

**Plus:**
- Testing: 4 hours
- Documentation: 2 hours
- **Grand Total:** ~26 hours (~3 working days)

---

## Recommendation

### Should You Remove Haystack?

**Yes, if:**
- You want full control over the RAG pipeline
- You're comfortable maintaining orchestration code
- You want to minimize dependencies
- You're not planning to use other Haystack components

**No, if:**
- You want to use Haystack's component ecosystem
- You prefer battle-tested framework code
- You value framework updates and community support
- Development time is more important than dependency count

### My Take

Given that we've already built 90% of the pipeline with custom components, **removing Haystack makes sense**. The orchestration code is straightforward (~200 lines), and we gain full control with minimal effort.

The migration is **low-risk** because:
1. Most code is already Haystack-independent
2. We can test both pipelines side-by-side
3. The custom orchestrator is simpler than Haystack's abstraction

---

## Next Steps

If you decide to proceed:

1. **Review this plan** - Any concerns or adjustments?
2. **Phase 1 first** - Build custom orchestrator alongside Haystack
3. **Test extensively** - Compare outputs between implementations
4. **Switch gradually** - Phase 2 → Phase 3 → Phase 4
5. **Document changes** - Update README and architecture docs

Let me know if you'd like me to start implementing Phase 1!
