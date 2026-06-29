"""Health check endpoint."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
import os

from config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_vectors: int
    knowledge_base_exists: bool
    api_keys_configured: int


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    vs = getattr(request.app.state, "vector_service", None)
    index_loaded = vs is not None and vs._index is not None
    index_vectors = vs._index.ntotal if index_loaded else 0

    return HealthResponse(
        status="ok",
        index_loaded=index_loaded,
        index_vectors=index_vectors,
        knowledge_base_exists=os.path.isfile(settings.knowledge_base_path),
        api_keys_configured=len(settings.gemini_api_keys),
    )
