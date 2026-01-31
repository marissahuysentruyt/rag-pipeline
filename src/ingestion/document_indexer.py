"""
Document indexer for design system documentation.
Supports both legacy Haystack embedders and new EmbeddingProvider interface.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.ingestion.document_processor import DocumentProcessor, DocumentChunk
from src.embedding.providers.base import EmbeddingProvider
from src.embedding.providers.sentence_transformers import SentenceTransformersProvider
from src.embedding.providers import EmbeddingConfig

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Indexes processed documents into Chroma vector store.

    Supports two modes:
    1. Legacy mode: Uses Haystack SentenceTransformersDocumentEmbedder (default)
    2. Provider mode: Uses EmbeddingProvider interface for modular architecture
    """

    def __init__(
        self,
        collection_name: str = "design_system_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_path: str = "./data/chroma_db",
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        """
        Initialize document indexer.

        Args:
            collection_name: Name of the Chroma collection
            embedding_model: Name of the sentence transformers model (used if embedding_provider is None)
            persist_path: Path to persist Chroma database
            embedding_provider: Optional EmbeddingProvider instance for modular architecture
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_path = persist_path
        self.embedding_provider = embedding_provider
        self.use_provider = embedding_provider is not None

        # Initialize document store
        logger.info(f"Initializing Chroma document store at {persist_path}")
        self.document_store = ChromaDocumentStore(
            collection_name=collection_name,
            persist_path=persist_path
        )

        if self.use_provider:
            # Use new modular architecture
            logger.info(f"Using EmbeddingProvider: {type(embedding_provider).__name__}")
            self.embedder = None
            self.indexing_pipeline = None
        else:
            # Use legacy Haystack embedder
            logger.info(f"Initializing legacy Haystack embedder: {embedding_model}")
            self.embedder = SentenceTransformersDocumentEmbedder(
                model=embedding_model
            )
            self.embedder.warm_up()

            # Create indexing pipeline
            self.indexing_pipeline = Pipeline()
            self.indexing_pipeline.add_component("embedder", self.embedder)

    def chunk_to_haystack_document(self, chunk: DocumentChunk) -> Document:
        """
        Convert a DocumentChunk to a Haystack Document.

        Args:
            chunk: DocumentChunk object

        Returns:
            Haystack Document with content and metadata
        """
        # Prepare metadata
        metadata = {
            **chunk.metadata,
            "chunk_type": chunk.chunk_type,
            "heading": chunk.heading
        }

        # Clean metadata - remove None values and ensure JSON serializable
        clean_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                # Convert to string if not a basic type
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)

        return Document(
            content=chunk.content,
            meta=clean_metadata
        )

    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
        batch_size: int = 100
    ) -> int:
        """
        Index a list of document chunks.

        Args:
            chunks: List of DocumentChunk objects
            duplicate_policy: How to handle duplicate documents
            batch_size: Number of documents to process in each batch

        Returns:
            Number of documents indexed
        """
        if not chunks:
            logger.warning("No chunks to index")
            return 0

        logger.info(f"Converting {len(chunks)} chunks to Haystack documents...")
        documents = [self.chunk_to_haystack_document(chunk) for chunk in chunks]

        logger.info(f"Generating embeddings and indexing {len(documents)} documents...")

        if self.use_provider:
            # Use new EmbeddingProvider interface
            return self._index_with_provider(documents, duplicate_policy, batch_size)
        else:
            # Use legacy Haystack embedder
            return self._index_with_haystack(documents, duplicate_policy, batch_size)

    def _index_with_haystack(
        self,
        documents: List[Document],
        duplicate_policy: DuplicatePolicy,
        batch_size: int
    ) -> int:
        """
        Index documents using legacy Haystack embedder.

        Args:
            documents: List of Haystack Document objects
            duplicate_policy: How to handle duplicate documents
            batch_size: Number of documents to process in each batch

        Returns:
            Number of documents indexed
        """
        total_indexed = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")

            # Generate embeddings
            embedded_docs = self.embedder.run(documents=batch)
            docs_with_embeddings = embedded_docs["documents"]

            # Write to document store
            self.document_store.write_documents(
                documents=docs_with_embeddings,
                policy=duplicate_policy
            )

            total_indexed += len(batch)
            logger.info(f"Indexed {total_indexed}/{len(documents)} documents")

        logger.info(f"Successfully indexed {total_indexed} documents")
        return total_indexed

    def _index_with_provider(
        self,
        documents: List[Document],
        duplicate_policy: DuplicatePolicy,
        batch_size: int
    ) -> int:
        """
        Index documents using EmbeddingProvider interface.

        Args:
            documents: List of Haystack Document objects
            duplicate_policy: How to handle duplicate documents
            batch_size: Number of documents to process in each batch

        Returns:
            Number of documents indexed
        """
        total_indexed = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")

            # Extract text content from documents
            texts = [doc.content for doc in batch]

            # Generate embeddings using provider
            embeddings = self.embedding_provider.embed_batch(texts)

            # Attach embeddings to documents
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding

            # Write to document store
            self.document_store.write_documents(
                documents=batch,
                policy=duplicate_policy
            )

            total_indexed += len(batch)
            logger.info(f"Indexed {total_indexed}/{len(documents)} documents")

        logger.info(f"Successfully indexed {total_indexed} documents")
        return total_indexed

    def get_stats(self) -> dict:
        """Get statistics about the indexed documents."""
        count = self.document_store.count_documents()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "persist_path": self.persist_path
        }

    def clear_index(self):
        """Clear all documents from the index."""
        logger.warning(f"Clearing all documents from collection: {self.collection_name}")
        # Delete all documents by filtering with empty filter
        all_docs = self.document_store.filter_documents()
        if all_docs:
            doc_ids = [doc.id for doc in all_docs]
            self.document_store.delete_documents(doc_ids)
        logger.info(f"Cleared {len(all_docs) if all_docs else 0} documents")

    def convert_chunks_to_documents(self, chunks: List) -> List[DocumentChunk]:
        """
        Convert Chunk objects to DocumentChunk objects for indexing.
        
        Args:
            chunks: List of Chunk objects from processing
            
        Returns:
            List of DocumentChunk objects
        """
        doc_chunks = []
        
        for chunk in chunks:
            # Map ChunkType to string
            chunk_type_str = "code" if hasattr(chunk, 'chunk_type') and chunk.chunk_type.name == "CODE" else "text"
            
            doc_chunk = DocumentChunk(
                content=chunk.content,
                metadata=chunk.metadata or {},
                chunk_type=chunk_type_str,
                heading=chunk.heading
            )
            doc_chunks.append(doc_chunk)
        
        logger.info(f"Converted {len(doc_chunks)} chunks to DocumentChunk objects")
        return doc_chunks

    def index_chunks_from_pipeline(self, chunks: List, batch_size: int = 10) -> int:
        """
        Index chunks from the processing pipeline.
        
        Args:
            chunks: List of Chunk objects from processing
            batch_size: Batch size for processing
            
        Returns:
            Number of chunks indexed
        """
        # Convert chunks to DocumentChunk objects
        doc_chunks = self.convert_chunks_to_documents(chunks)
        
        # Index the converted chunks
        return self.index_chunks(doc_chunks, batch_size=batch_size)

    def get_sample_document_info(self, chunks: List[DocumentChunk]) -> dict:
        """
        Get information about a sample document for display.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dictionary with sample document information
        """
        if not chunks:
            return {}
        
        sample = chunks[0]
        
        return {
            "content_length": len(sample.content),
            "chunk_type": sample.chunk_type,
            "heading": sample.heading or "None",
            "component": sample.metadata.get("component", "N/A") if sample.metadata else "N/A",
            "category": sample.metadata.get("category", "N/A") if sample.metadata else "N/A",
            "title": sample.metadata.get("title", "N/A") if sample.metadata else "N/A"
        }


def main():
    """Run the indexing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Process documents
    logger.info("Processing documents...")
    processor = DocumentProcessor(
        min_chunk_size=200,
        max_chunk_size=1500
    )
    chunks = processor.process_directory(Path("data/raw/crawled"))

    if not chunks:
        logger.error("No chunks created from documents")
        return

    logger.info(f"Created {len(chunks)} chunks")

    # Index documents
    logger.info("Indexing documents...")
    indexer = DocumentIndexer(
        collection_name="design_system_docs",
        persist_path="./data/chroma_db"
    )

    # Clear existing index (optional - comment out to append)
    # indexer.clear_index()

    indexed_count = indexer.index_chunks(chunks, batch_size=50)

    # Print stats
    stats = indexer.get_stats()
    logger.info(f"Indexing complete!")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
