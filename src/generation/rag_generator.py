"""
Generation module for RAG-based response generation.

This module provides high-level interfaces for generating responses
using retrieved context and LLM providers.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.query.rag_pipeline import DesignSystemRAG
from src.retrieval import ChromaRetriever, RetrievalConfig
from src.embedding.providers import SentenceTransformersProvider, EmbeddingConfig
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    High-level RAG generator for contextual responses.
    
    Combines retrieval and generation to provide answers based on
    retrieved documentation context.
    """
    
    def __init__(
        self,
        collection_name: str = "golden_demo",
        persist_path: str = "./data/demo_chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "claude-sonnet-4-5-20250929",
        top_k: int = 3
    ):
        """
        Initialize RAG generator.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_path: Path to Chroma database
            embedding_model: Sentence transformers model
            llm_model: Claude model to use
            top_k: Number of documents to retrieve
        """
        self.collection_name = collection_name
        self.persist_path = persist_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        
        self.rag_pipeline = None
        self.retriever = None
        
    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline and retriever.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize RAG pipeline
            self.rag_pipeline = DesignSystemRAG(
                collection_name=self.collection_name,
                persist_path=self.persist_path,
                embedding_model=self.embedding_model,
                llm_model=self.llm_model,
                top_k=self.top_k
            )
            
            # Initialize retriever for context building
            document_store = ChromaDocumentStore(
                collection_name=self.collection_name,
                persist_path=self.persist_path
            )
            
            embedding_config = EmbeddingConfig(
                model_name=self.embedding_model,
                dimensions=384
            )
            embedding_provider = SentenceTransformersProvider(embedding_config)
            embedding_provider.load_model()
            
            self.retriever = ChromaRetriever(
                config=RetrievalConfig(top_k=self.top_k),
                document_store=document_store,
                embedding_provider=embedding_provider
            )
            
            logger.info("RAG generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG generator: {e}")
            return False
    
    def check_api_key(self) -> bool:
        """
        Check if Anthropic API key is configured.
        
        Returns:
            True if API key is available, False otherwise
        """
        if self.rag_pipeline:
            return self.rag_pipeline.check_api_key()
        return False
    
    def check_index_status(self) -> Dict[str, Any]:
        """
        Check the status of the indexed documents.
        
        Returns:
            Dictionary with index status information
        """
        if not self.retriever:
            return {"has_documents": False, "error": "Retriever not initialized"}
        
        return self.retriever.check_index_status()
    
    def generate_response(
        self,
        query: str,
        include_context: bool = True,
        max_context_docs: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a response for the given query.
        
        Args:
            query: User query
            include_context: Whether to include context in response
            max_context_docs: Maximum number of context documents to include
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.rag_pipeline:
            return {"error": "RAG pipeline not initialized"}
        
        if not self.check_api_key():
            return {"error": "Anthropic API key not configured"}
        
        try:
            # Generate response using RAG pipeline
            result = self.rag_pipeline.query(query)
            
            response = {
                "query": query,
                "answer": result["answer"],
                "documents": result["documents"],
                "metadata": result["metadata"]
            }
            
            # Add context if requested
            if include_context and self.retriever:
                retrieval_results = self.retriever.retrieve(query)
                context = self.rag_pipeline.build_context_from_documents(
                    retrieval_results.documents, 
                    max_context_docs
                )
                response["context"] = context
            
            # Add formatted sources
            response["sources"] = self.rag_pipeline.format_sources_for_display(result["documents"])
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e)}
    
    def get_generation_stats(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about the generation response.
        
        Args:
            response: Response dictionary from generate_response
            
        Returns:
            Dictionary with generation statistics
        """
        if "error" in response:
            return {"error": response["error"]}
        
        metadata = response.get("metadata", {})
        documents = response.get("documents", [])
        
        return {
            "model": metadata.get("model", "unknown"),
            "documents_retrieved": metadata.get("num_documents", len(documents)),
            "query_length": len(response.get("query", "")),
            "answer_length": len(response.get("answer", "")),
            "has_context": "context" in response,
            "sources_count": len(response.get("sources", []))
        }
