from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..models.schemas import AnalyzeRequest, AnalyzeResponse
from ..services.analyzer import analyze_image

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("", response_model=AnalyzeResponse, summary="Analyze a moment image")
async def analyze(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """
    Run the full AI analysis pipeline on a photo URL:
    - EXIF extraction (GPS, timestamp, camera)
    - Vehicle detection (YOLOv8)
    - License plate OCR (EasyOCR)
    - Image embedding (CLIP ViT-B/32)
    - Scene tag classification (CLIP zero-shot)

    Returns structured JSON ready to be stored in the NestJS `moments.ai_analysis` column.
    """
    return analyze_image(request, settings)
