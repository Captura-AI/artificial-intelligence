import anyio
from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..models.schemas import AnalyzeRequest, AnalyzeResponse
from ..services.analyzer import analyze_image

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Defense-in-depth cap on concurrent heavy pipeline runs. The backend BullMQ
# queue is the primary throttle, but this protects the single-process model
# service from any parallel caller (direct calls, retries, multiple producers).
# Bounded to settings.max_concurrent_analyses; created once at import.
_analysis_semaphore = anyio.Semaphore(get_settings().max_concurrent_analyses)


@router.post("", response_model=AnalyzeResponse, summary="Analyze a moment image")
async def analyze(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """
    Run the full AI analysis pipeline on a photo URL:
    - EXIF extraction (GPS, timestamp, camera)
    - Vehicle detection (YOLOv8)
    - License plate detection + character reading (YOLOv8)
    - Image embedding (CLIP ViT-B/32)
    - Scene tag classification (CLIP zero-shot)

    Returns structured JSON ready to be stored in the NestJS `moments.ai_analysis` column.

    The CPU/GPU-bound pipeline is run in a worker thread under a concurrency
    semaphore: callers above the limit wait their turn instead of piling onto
    the models, and the event loop stays free to serve /health.
    """
    async with _analysis_semaphore:
        return await anyio.to_thread.run_sync(analyze_image, request, settings)
