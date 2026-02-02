"""
RAG query pipeline for design system documentation.
Retrieves relevant chunks and generates responses using Claude.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

logger = logging.getLogger(__name__)


class DesignSystemRAG:
    """RAG pipeline for querying design system documentation."""

    def __init__(
        self,
        collection_name: str = "design_system_docs",
        persist_path: str = "./data/chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "claude-sonnet-4-5-20250929",
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.

        Args:
            collection_name: Name of the Chroma collection
            persist_path: Path to Chroma database
            embedding_model: Sentence transformers model for embeddings
            llm_model: Claude model to use
            top_k: Number of documents to retrieve
        """
        self.collection_name = collection_name
        self.persist_path = persist_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k

        # Verify API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        # Initialize components
        logger.info("Initializing RAG pipeline components...")
        self._initialize_components()
        self._build_pipeline()

    def _initialize_components(self):
        """Initialize pipeline components."""
        # Document store
        logger.info(f"Connecting to Chroma at {self.persist_path}")
        self.document_store = ChromaDocumentStore(
            collection_name=self.collection_name,
            persist_path=self.persist_path
        )

        # Check document count
        doc_count = self.document_store.count_documents()
        logger.info(f"Connected to document store with {doc_count} documents")

        if doc_count == 0:
            raise ValueError(
                f"No documents found in collection '{self.collection_name}'. "
                "Please run the indexer first."
            )

        # Text embedder for queries
        logger.info(f"Loading embedder: {self.embedding_model}")
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=self.embedding_model
        )
        self.text_embedder.warm_up()

        # Retriever
        self.retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k
        )

        # Prompt builder
        prompt_template = [ChatMessage.from_system(
            "You are an expert assistant helping developers use Golden design system components and guidelines."
        )]

        self.prompt_builder = ChatPromptBuilder(
            template=prompt_template + [ChatMessage.from_user("""Use the following documentation excerpts to answer my question. Each excerpt includes metadata about the source.

Context:
{% for doc in documents %}
---
Source: {{ doc.meta.title or doc.meta.url }}
{% if doc.meta.heading %}Section: {{ doc.meta.heading }}{% endif %}

{{ doc.content }}
{% endfor %}

Question: {{ query }}

Instructions:
- Provide a clear, accurate answer based on the documentation above
- If referencing specific components or patterns, mention them by name
- Include code examples from the documentation when relevant
- If the documentation doesn't contain enough information, say so (i.e. say "I don't know" as opposed to inferring)
- Cite sources by mentioning the component or page name""")]
        )

        # LLM
        logger.info(f"Initializing Claude: {self.llm_model}")
        self.llm = AnthropicChatGenerator(
            model=self.llm_model,
            generation_kwargs={
                "max_tokens": 2000,
                "temperature": 0.3
            }
        )

    def _build_pipeline(self):
        """Build the RAG pipeline."""
        logger.info("Building RAG pipeline...")
        self.pipeline = Pipeline()

        # Add components
        self.pipeline.add_component("text_embedder", self.text_embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)

        # Connect components
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

        logger.info("RAG pipeline built successfully")

    def query(self, question: str, filters: Optional[Dict] = None) -> Dict:
        """
        Query the design system documentation.

        Args:
            question: User's question
            filters: Optional metadata filters (e.g., {"domain": "spectrum.adobe.com"})

        Returns:
            Dict with 'answer', 'documents', and 'metadata'
        """
        logger.info(f"Processing query: {question[:100]}...")

        try:
            # Run pipeline
            result = self.pipeline.run({
                "text_embedder": {"text": question},
                "retriever": {"filters": filters} if filters else {},
                "prompt_builder": {"query": question}
            })

            # Extract results - get documents from retriever before it was consumed by prompt_builder
            # We need to run retrieval separately to get documents
            query_embedding = self.text_embedder.run(text=question)
            retrieval_result = self.retriever.run(
                query_embedding=query_embedding["embedding"],
                filters=filters
            )
            documents = retrieval_result["documents"]

            # Extract answer from LLM
            answer = result["llm"]["replies"][0].text

            logger.info(f"Retrieved {len(documents)} documents, generated response")

            return {
                "answer": answer,
                "documents": documents,
                "metadata": {
                    "query": question,
                    "num_documents": len(documents),
                    "model": self.llm_model,
                    "filters": filters
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise

    def get_relevant_docs(self, question: str, filters: Optional[Dict] = None) -> List:
        """
        Get relevant documents without generating a response.

        Args:
            question: User's question
            filters: Optional metadata filters

        Returns:
            List of relevant Document objects
        """
        # Embed query
        embedded = self.text_embedder.run(text=question)

        # Retrieve documents
        retriever_params = {"query_embedding": embedded["embedding"]}
        if filters:
            retriever_params["filters"] = filters

        result = self.retriever.run(**retriever_params)
        return result["documents"]

    def build_context_from_documents(self, documents: List, max_docs: int = 3) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved Document objects
            max_docs: Maximum number of documents to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents[:max_docs], 1):
            # Handle both Haystack Document (meta) and RetrievedDocument (metadata)
            if hasattr(doc, 'meta'):
                meta = doc.meta
            else:
                meta = doc.metadata
                
            source = meta.get("component", meta.get("title", "Unknown"))
            heading = meta.get("heading", "")
            header = f"Source: {source}"
            if heading:
                header += f" > {heading}"
            context_parts.append(f"--- {header} ---\n{doc.content}")
        
        return "\n\n".join(context_parts)

    def check_api_key(self) -> bool:
        """
        Check if Anthropic API key is configured.
        
        Returns:
            True if API key is available, False otherwise
        """
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def format_sources_for_display(self, documents: List) -> List[Dict[str, str]]:
        """
        Format sources for display.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of formatted source information
        """
        sources = []
        for doc in documents:
            # Handle both Haystack Document (meta) and RetrievedDocument (metadata)
            if hasattr(doc, 'meta'):
                meta = doc.meta
            else:
                meta = doc.metadata
                
            source = meta.get("component", meta.get("title", "Unknown"))
            score = f"{doc.score:.3f}" if doc.score else "N/A"
            sources.append({
                "source": source,
                "score": score,
                "heading": meta.get("heading", ""),
                "url": meta.get("url", "")
            })
        return sources


def main():
    """Test the RAG pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize RAG
    rag = DesignSystemRAG(top_k=5)

    # Test queries
    test_queries = [
        "How do I use a button component in React Spectrum?",
        "What are the color guidelines for the Spectrum design system?",
        "Show me an example of a ComboBox with dynamic data"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = rag.query(query)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources ({result['metadata']['num_documents']} documents):")
        for i, doc in enumerate(result['documents'], 1):
            print(f"{i}. {doc.meta.get('title', 'Unknown')} - {doc.meta.get('url', 'N/A')}")
            if doc.score:
                print(f"   Relevance: {doc.score:.3f}")


if __name__ == "__main__":
    main()
