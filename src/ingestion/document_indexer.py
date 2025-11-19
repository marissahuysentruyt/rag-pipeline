"""
Document indexer for design system documentation.
Uses Haystack to generate embeddings and store in Chroma.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.ingestion.document_processor import DocumentProcessor, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Indexes processed documents into Chroma vector store."""

    def __init__(
        self,
        collection_name: str = "design_system_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_path: str = "./data/chroma_db"
    ):
        """
        Initialize document indexer.

        Args:
            collection_name: Name of the Chroma collection
            embedding_model: Name of the sentence transformers model
            persist_path: Path to persist Chroma database
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_path = persist_path

        # Initialize document store
        logger.info(f"Initializing Chroma document store at {persist_path}")
        self.document_store = ChromaDocumentStore(
            collection_name=collection_name,
            persist_path=persist_path
        )

        # Initialize embedder
        logger.info(f"Initializing embedder: {embedding_model}")
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

        # Process in batches to avoid memory issues
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
