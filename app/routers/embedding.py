from fastapi import APIRouter, Depends, HTTPException

from ..config import Settings, get_settings
from ..models.schemas import TextEmbeddingRequest, TextEmbeddingResponse
from ..services.clip_embedder import get_text_embedding

router = APIRouter(prefix="/embed", tags=["Embedding"])


@router.post("/text", response_model=TextEmbeddingResponse, summary="Generate a text embedding")
async def embed_text(
    request: TextEmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> TextEmbeddingResponse:
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    embedding = get_text_embedding(query, settings.clip_model_name)

    if not embedding:
        raise HTTPException(status_code=503, detail="Text embedding model is unavailable.")

    return TextEmbeddingResponse(
        embedding=embedding,
        model=settings.clip_model_name,
        query=query,
        vector_dimension=len(embedding),
    )
