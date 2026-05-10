from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db.database import init_db
from .models.schemas import HealthResponse
from .routers import analysis
from .routers import plate as plate_router
from .services.clip_embedder import is_model_ready as clip_ready
from .services.plate_detector import is_model_ready as platdetect_ready
from .services.plate_text_reader import is_model_ready as platreader_ready
from .services.vehicle_detector import is_model_ready as yolo_ready

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise SQLite database
    init_db(settings.database_url)

    # Warm up models on startup so first request is not slow
    print("Warming up AI models...")
    yolo_ready(settings.yolo_model_path)
    clip_ready(settings.clip_model_name)
    platdetect_ready(settings.platdetect_model_path)
    platreader_ready(settings.platreader_model_path)
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
app.include_router(plate_router.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(
        models_loaded={
            "clip": clip_ready(settings.clip_model_name),
            "yolo": yolo_ready(settings.yolo_model_path),
            "platdetect": platdetect_ready(settings.platdetect_model_path),
            "platreader": platreader_ready(settings.platreader_model_path),
        },
        status="ok",
        version=settings.app_version,
    )
