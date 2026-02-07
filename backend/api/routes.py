"""API routes for the RAG backend."""

import re

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.api.models import (
    ConfigResponse,
    DeleteResponse,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from backend.services.rag_service import get_rag_service

router = APIRouter(prefix="/api", tags=["RAG API"])


def sanitize_error(error: Exception) -> str:
    """Remove sensitive information from error messages.

    Removes absolute file paths from error messages to prevent
    exposing filesystem structure to API clients.
    """
    msg = str(error)
    # Remove absolute paths, keep only filename
    # Matches /path/to/file.ext and replaces with just file.ext
    msg = re.sub(r"/[^\s:]+/([^/\s:]+)", r"\1", msg)
    return msg


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the status of the RAG service including Ollama availability
    and document statistics.
    """
    service = get_rag_service()
    return service.get_health()


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get configuration settings.

    Returns the current configuration including default LLM provider,
    retrieval settings, and chunking parameters.
    """
    service = get_rag_service()
    return service.get_config()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base.

    Submit a question and get an answer based on indexed documents.
    The response includes the answer, source chunks used, and metadata.
    """
    service = get_rag_service()

    try:
        result = service.query(
            question=request.query,
            llm_provider=request.llm_provider,
            top_k=request.top_k,
            temperature=request.temperature,
        )

        # Convert to response model
        return QueryResponse(
            answer=result["answer"],
            sources=[
                {
                    "file": s["file"],
                    "chunk_id": s["chunk_id"],
                    "content": s["content"],
                    "similarity": s["similarity"],
                }
                for s in result["sources"]
            ]
            if request.show_sources
            else [],
            metadata=result["metadata"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=sanitize_error(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {sanitize_error(e)}")


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    force: bool = Query(False, description="Re-index if document already exists"),
):
    """
    Upload and index a document.

    Supported formats: PDF (.pdf), Markdown (.md), HTML (.html, .htm),
    Plain text (.txt).
    """
    service = get_rag_service()

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    supported = service.get_supported_extensions()
    if "." in file.filename:
        file_ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    else:
        file_ext = ""

    if file_ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. "
            f"Supported formats: {', '.join(supported)}",
        )

    try:
        result = service.upload_document(
            file=file.file,
            filename=file.filename,
            force=force,
        )
        return UploadResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {sanitize_error(e)}")


@router.get(
    "/documents",
    response_model=DocumentListResponse,
)
async def list_documents():
    """
    List all indexed documents.

    Returns information about each document including filename,
    file type, and chunk count.
    """
    service = get_rag_service()
    result = service.list_documents()

    return DocumentListResponse(
        documents=[
            {
                "id": d["id"],
                "filename": d["filename"],
                "file_type": d["file_type"],
                "chunk_count": d["chunk_count"],
                "indexed_at": d["indexed_at"],
            }
            for d in result["documents"]
        ],
        total_documents=result["total_documents"],
        total_chunks=result["total_chunks"],
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_document(document_id: str):
    """
    Delete a document from the knowledge base.

    The document_id should be the filename as shown in GET /api/documents.
    """
    service = get_rag_service()

    try:
        result = service.delete_document(document_id)
        return DeleteResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=sanitize_error(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {sanitize_error(e)}")
