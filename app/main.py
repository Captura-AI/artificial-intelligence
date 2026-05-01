from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models.schemas import HealthResponse
from .routers import analysis
from .services.clip_embedder import is_model_ready as clip_ready
from .services.plate_reader import is_model_ready as ocr_ready
from .services.vehicle_detector import is_model_ready as yolo_ready

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models on startup so first request is not slow
    print("Warming up AI models...")
    yolo_ready(settings.yolo_model_path)
    ocr_ready(settings.ocr_languages)
    clip_ready(settings.clip_model_name)
    print("Models ready.")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI microservice for Captura — extracts vehicle type, license plate, "
        "GPS metadata, and CLIP embeddings from street photography."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)

app.include_router(analysis.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(
        models_loaded={
            "clip": clip_ready(settings.clip_model_name),
            "easyocr": ocr_ready(settings.ocr_languages),
            "yolo": yolo_ready(settings.yolo_model_path),
        },
        status="ok",
        version=settings.app_version,
    )
