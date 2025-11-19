"""
FastAPI server for RAG pipeline.

Provides REST API endpoints for querying design system documentation.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.query.rag_pipeline import DesignSystemRAG
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.document_indexer import DocumentIndexer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Design System RAG API",
    description="Retrieval-Augmented Generation API for design system documentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance (initialized on startup)
rag_instance: Optional[DesignSystemRAG] = None

# Track refresh status
refresh_status = {
    "last_refresh": None,
    "in_progress": False,
    "last_error": None
}


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(..., description="The question to ask", min_length=1)
    domain: Optional[str] = Field(None, description="Filter results by domain")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)


class DocumentSource(BaseModel):
    """Model for a source document."""
    title: str
    url: str
    score: float
    content: str


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str
    sources: List[DocumentSource]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    version: str
    database_connected: bool
    api_key_configured: bool
    total_documents: Optional[int] = None


class StatsResponse(BaseModel):
    """Response model for /stats endpoint."""
    total_documents: int
    collection_name: str
    embedding_model: str
    llm_model: str
    persist_path: str
    last_refresh: Optional[str] = None


class RefreshResponse(BaseModel):
    """Response model for /refresh endpoint."""
    message: str
    status: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on server startup."""
    global rag_instance

    logger.info("Starting up RAG API server...")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not found in environment")
        raise RuntimeError("ANTHROPIC_API_KEY must be set")

    try:
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_instance = DesignSystemRAG(
            collection_name="design_system_docs",
            persist_path="./data/chroma_db"
        )
        logger.info("RAG pipeline initialized successfully")

        # Log initial stats
        stats = rag_instance.document_store.count_documents()
        logger.info(f"Loaded {stats} documents from index")

    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    logger.info("Shutting down RAG API server...")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Design System RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Query the documentation",
            "/health": "GET - Health check",
            "/stats": "GET - Index statistics",
            "/refresh": "POST - Refresh documentation index"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API service, including database
    connectivity and configuration status.
    """
    try:
        # Check if RAG instance is initialized
        if rag_instance is None:
            return HealthResponse(
                status="unhealthy",
                version="1.0.0",
                database_connected=False,
                api_key_configured=bool(os.getenv("ANTHROPIC_API_KEY"))
            )

        # Check database connectivity
        try:
            doc_count = rag_instance.document_store.count_documents()
            db_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            doc_count = None
            db_connected = False

        # Check API key
        api_key_configured = bool(os.getenv("ANTHROPIC_API_KEY"))

        # Overall health status
        is_healthy = db_connected and api_key_configured

        return HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            version="1.0.0",
            database_connected=db_connected,
            api_key_configured=api_key_configured,
            total_documents=doc_count
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["system"])
async def get_stats():
    """
    Get index statistics.

    Returns statistics about the document index, including total document count,
    collection name, embedding model, and LLM model.
    """
    try:
        if rag_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )

        # Get document count
        doc_count = rag_instance.document_store.count_documents()

        return StatsResponse(
            total_documents=doc_count,
            collection_name=rag_instance.collection_name,
            embedding_model=rag_instance.embedding_model,
            llm_model=rag_instance.llm_model,
            persist_path=rag_instance.persist_path,
            last_refresh=refresh_status["last_refresh"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_documentation(request: QueryRequest):
    """
    Query the design system documentation.

    Uses RAG (Retrieval-Augmented Generation) to answer questions about
    design system documentation. Retrieves relevant documents and generates
    a response using Claude.

    Args:
        request: Query request with question, optional domain filter, and top_k

    Returns:
        QueryResponse with answer, source documents, and metadata
    """
    try:
        if rag_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )

        logger.info(f"Processing query: {request.question}")

        # Build filters if domain specified
        filters = None
        if request.domain:
            filters = {
                "field": "domain",
                "operator": "==",
                "value": request.domain
            }

        # Execute query
        try:
            # Update top_k if different from default
            if request.top_k != rag_instance.top_k:
                rag_instance.top_k = request.top_k
                rag_instance.retriever.top_k = request.top_k

            result = rag_instance.query(
                question=request.question,
                filters=filters
            )

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query execution failed: {str(e)}"
            )

        # Format response
        sources = []
        for doc in result["documents"]:
            sources.append(DocumentSource(
                title=doc.meta.get("title", "Untitled"),
                url=doc.meta.get("url", ""),
                score=doc.score if hasattr(doc, 'score') else 0.0,
                content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            ))

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            metadata=result["metadata"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


async def refresh_documentation_index():
    """
    Background task to refresh the documentation index.

    Recrawls documentation sources and re-indexes documents.
    """
    global rag_instance, refresh_status

    try:
        refresh_status["in_progress"] = True
        refresh_status["last_error"] = None

        logger.info("Starting documentation refresh...")

        # Step 1: Crawl documentation
        logger.info("Crawling documentation sources...")
        result = subprocess.run(
            ["python", "src/ingestion/crawl_docs.py", "crawl"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Crawling failed: {result.stderr}")
        logger.info(f"Crawl output: {result.stdout}")

        # Step 2: Process documents
        logger.info("Processing documents...")
        processor = DocumentProcessor(
            min_chunk_size=200,
            max_chunk_size=1500
        )
        chunks = processor.process_directory(Path("data/raw/crawled"))
        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Index documents
        logger.info("Indexing documents...")
        indexer = DocumentIndexer(
            collection_name="design_system_docs",
            persist_path="./data/chroma_db"
        )

        # Clear existing index
        indexer.clear_index()

        # Index new chunks
        indexed_count = indexer.index_chunks(chunks, batch_size=50)
        logger.info(f"Indexed {indexed_count} documents")

        # Step 4: Reinitialize RAG instance
        logger.info("Reinitializing RAG pipeline...")
        rag_instance = DesignSystemRAG(
            collection_name="design_system_docs",
            persist_path="./data/chroma_db"
        )

        # Update refresh status
        refresh_status["last_refresh"] = datetime.utcnow().isoformat()
        refresh_status["in_progress"] = False

        logger.info("Documentation refresh completed successfully")

    except Exception as e:
        logger.error(f"Documentation refresh failed: {e}")
        refresh_status["last_error"] = str(e)
        refresh_status["in_progress"] = False
        raise


@app.post("/refresh", response_model=RefreshResponse, tags=["system"])
async def refresh_index(background_tasks: BackgroundTasks):
    """
    Refresh the documentation index.

    Triggers a background task to recrawl documentation sources and
    re-index all documents. This is an asynchronous operation that
    runs in the background.

    Returns immediately with a status message. Check /stats endpoint
    for last_refresh timestamp to see when the refresh completed.
    """
    try:
        # Check if refresh is already in progress
        if refresh_status["in_progress"]:
            return RefreshResponse(
                message="Refresh already in progress",
                status="in_progress",
                timestamp=datetime.utcnow().isoformat()
            )

        # Add refresh task to background
        background_tasks.add_task(refresh_documentation_index)

        return RefreshResponse(
            message="Documentation refresh started",
            status="started",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to start refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start refresh: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
