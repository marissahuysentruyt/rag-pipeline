

# Embedding Providers Guide

The RAG pipeline supports multiple embedding providers, allowing you to choose the best option for your use case based on cost, performance, and deployment requirements.

## Available Providers

### 1. Sentence Transformers (Local)

**Best for:** Development, privacy-sensitive deployments, no API costs

```python
from src.embedding.factory import create_embedding_provider

provider = create_embedding_provider(
    "sentence-transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    dimensions=384
)
provider.load_model()
embedding = provider.embed_text("Your text here")
```

**Pros:**
- ✅ Free and open source
- ✅ Runs locally (no API calls)
- ✅ No rate limits
- ✅ Privacy-friendly (data stays local)
- ✅ Many models available on HuggingFace

**Cons:**
- ❌ Requires local compute resources
- ❌ Slower for large batches without GPU
- ❌ Model loading time on first use

**Models:**
- `all-MiniLM-L6-v2` - 384 dim, fast, good quality (default)
- `all-mpnet-base-v2` - 768 dim, slower, higher quality
- `multi-qa-MiniLM-L6-cos-v1` - 384 dim, optimized for Q&A

---

### 2. OpenAI Embeddings (API)

**Best for:** Production deployments, high-quality embeddings, minimal setup

```python
from src.embedding.factory import create_embedding_provider

provider = create_embedding_provider(
    "openai",
    model_name="text-embedding-3-small",
    api_key="sk-...",  # Or use ${OPENAI_API_KEY}
    dimensions=1536
)
provider.load_model()
embedding = provider.embed_text("Your text here")
```

**Pros:**
- ✅ High-quality embeddings
- ✅ Fast inference (no local compute needed)
- ✅ Scalable
- ✅ Multiple model options

**Cons:**
- ❌ Requires API key and internet connection
- ❌ Pay per use (~$0.00002 per 1K tokens)
- ❌ Rate limits (5000 RPM for most models)
- ❌ Data sent to OpenAI servers

**Models:**
- `text-embedding-3-small` - 1536 dim, $0.00002/1K tokens (recommended)
- `text-embedding-3-large` - 3072 dim, $0.00013/1K tokens, higher quality
- `text-embedding-ada-002` - 1536 dim, legacy model

---

## Factory Pattern Usage

### Basic Usage

```python
from src.embedding.factory import create_embedding_provider

# Sentence Transformers (local)
provider = create_embedding_provider("sentence-transformers")

# OpenAI (requires API key)
provider = create_embedding_provider(
    "openai",
    api_key="sk-..."
)
```

### From Configuration File

**config.yaml:**
```yaml
embedding:
  provider: openai
  model_name: text-embedding-3-small
  dimensions: 1536
  api_key: ${OPENAI_API_KEY}  # Environment variable
  batch_size: 100
  normalize: true
```

**Python:**
```python
from src.embedding.factory import EmbeddingProviderFactory

provider = EmbeddingProviderFactory.create_from_yaml("config.yaml")
provider.load_model()
```

### From Dictionary

```python
from src.embedding.factory import EmbeddingProviderFactory

config = {
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "dimensions": 1536
}

provider = EmbeddingProviderFactory.create_from_dict(config)
```

---

## Using with DocumentIndexer

The DocumentIndexer supports both legacy Haystack mode and the new provider architecture:

### Option 1: Provider Mode (Recommended)

```python
from src.embedding.factory import create_embedding_provider
from src.ingestion.document_indexer import DocumentIndexer

# Create provider
provider = create_embedding_provider(
    "openai",
    api_key=os.getenv("OPENAI_API_KEY")
)
provider.load_model()

# Create indexer with provider
indexer = DocumentIndexer(
    collection_name="my_docs",
    persist_path="./data/chroma_db",
    embedding_provider=provider  # Use modular provider
)

# Index documents
indexer.index_chunks(chunks)
```

### Option 2: Legacy Mode (Still Supported)

```python
from src.ingestion.document_indexer import DocumentIndexer

# Create indexer without provider (uses Haystack)
indexer = DocumentIndexer(
    collection_name="my_docs",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    persist_path="./data/chroma_db"
)

# Index documents
indexer.index_chunks(chunks)
```

---

## Comparison

### Performance

Run the comparison script:
```bash
python examples/compare_embedding_providers.py --compare
```

**Typical Results:**

| Metric | Sentence Transformers | OpenAI |
|--------|----------------------|--------|
| Model | all-MiniLM-L6-v2 | text-embedding-3-small |
| Dimensions | 384 | 1536 |
| Load time | ~1-2s | ~0.1s (API test) |
| Single embed | ~0.5s | ~0.2s |
| Batch (100) | ~3s | ~2s |
| Cost per 1M tokens | Free | ~$20 |

### Use Case Recommendations

**Use Sentence Transformers when:**
- Budget is a concern
- Privacy/data sovereignty required
- Offline operation needed
- Low-moderate volume (<100K docs)

**Use OpenAI when:**
- Need highest quality embeddings
- High volume indexing
- Want minimal infrastructure
- Budget allows for API costs

---

## Advanced Usage

### Custom Batching

```python
provider = create_embedding_provider(
    "openai",
    api_key="sk-...",
    batch_size=50  # Process 50 texts per API call
)
```

### Context Manager

```python
with create_embedding_provider("sentence-transformers") as provider:
    embeddings = provider.embed_batch(texts)
    # Automatic cleanup after context
```

### Get Provider Information

```python
from src.embedding.factory import EmbeddingProviderFactory

# List all available providers
providers = EmbeddingProviderFactory.list_providers()
print(providers)  # {'sentence-transformers': True, 'openai': True}

# Get info about specific provider
info = EmbeddingProviderFactory.get_provider_info("openai")
print(info)
```

---

## Environment Variables

Create a `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Sentence Transformers (no API key needed)
# Just ensure the library is installed
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()

provider = create_embedding_provider(
    "openai",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

---

## Adding New Providers

To add support for a new embedding provider (e.g., Cohere, Vertex AI):

1. **Create provider class:**
   ```python
   # src/embedding/providers/cohere.py
   from .base import EmbeddingProvider, EmbeddingConfig

   class CohereEmbeddingProvider(EmbeddingProvider):
       def load_model(self):
           # Implementation

       def embed_text(self, text):
           # Implementation

       def embed_batch(self, texts):
           # Implementation
   ```

2. **Register in factory:**
   ```python
   # src/embedding/factory.py
   PROVIDERS = {
       "sentence-transformers": SentenceTransformersProvider,
       "openai": OpenAIEmbeddingProvider,
       "cohere": CohereEmbeddingProvider,  # Add here
   }
   ```

3. **Add tests:**
   ```python
   # tests/test_cohere_provider.py
   ```

That's it! The factory will automatically support the new provider.

---

## Troubleshooting

### OpenAI Provider Not Available

```
ValueError: Provider 'openai' is not available. Install required dependencies
```

**Solution:**
```bash
pip install openai
```

### API Key Errors

```
ModelLoadError: Authentication failed. Check your API key
```

**Solution:**
1. Verify API key is correct
2. Check environment variable is set: `echo $OPENAI_API_KEY`
3. Ensure `.env` file is loaded

### Rate Limit Errors

```
RateLimitError: OpenAI rate limit exceeded
```

**Solution:**
- Reduce batch size
- Add delays between requests
- Upgrade OpenAI plan for higher limits
- Switch to Sentence Transformers (no limits)

---

## Best Practices

1. **Development:** Use Sentence Transformers for fast iteration
2. **Production:** Evaluate OpenAI for quality vs. Sentence Transformers for cost
3. **Hybrid:** Use Sentence Transformers for bulk indexing, OpenAI for real-time queries
4. **Testing:** Write provider-agnostic tests using the base interface
5. **Configuration:** Use environment variables and config files, never hardcode API keys

---

## Cost Estimation

### OpenAI Costs

**Formula:** Cost = (Total tokens / 1000) × Price per 1K

**Example: Indexing 10,000 documents**
- Avg 100 tokens/doc = 1M tokens
- text-embedding-3-small: 1M × $0.00002 = $20
- text-embedding-3-large: 1M × $0.00013 = $130

**Query costs are typically negligible** (few tokens per query).

### Sentence Transformers Costs

- Software: Free (open source)
- Compute: Local CPU/GPU
- For 10K docs: ~10-30 minutes on CPU (one-time indexing)
