import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db.database import init_db
from .models.schemas import HealthResponse
from .routers import analysis
from .routers import embedding
from .routers import plate as plate_router
from .security import require_api_key
from .services.clip_embedder import is_model_ready as clip_ready
from .services.color_classifier import is_model_ready as color_ready
from .services.motor_type_detector import is_model_ready as motortype_ready
from .services.plate_detector import is_model_ready as platdetect_ready
from .services.plate_text_reader import is_model_ready as platreader_ready
from .services.vehicle_detector import is_model_ready as yolo_ready

settings = get_settings()

# ── Logging configuration ──────────────────────────────────────────────────────
# JSON-structured logs so lines can be parsed by log aggregators (Loki, CloudWatch,
# Datadog, etc.). Falls back to plain text if python-json-logger is not installed.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

try:
    from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]

    _formatter: logging.Formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
except ImportError:
    _formatter = logging.Formatter(_LOG_FORMAT)

_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler],
    force=True,
)

# Silence noisy third-party loggers
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("easyocr").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("clip").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: initialising database connection")
    init_db(settings.database_url, settings.db_pool_min_size, settings.db_pool_max_size)

    logger.info("startup: warming up AI models")
    yolo_ready(settings.yolo_model_path)
    clip_ready(settings.clip_model_name)
    platdetect_ready(settings.platdetect_model_path)
    platreader_ready(settings.platreader_model_path)
    motortype_ready(settings.motortype_model_path)
    color_ready(settings.color_model_path)
    logger.info("startup: all models ready")
    yield
    logger.info("shutdown: application stopping")


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
    allow_origins=settings.cors_allow_origins,
)

# Protect the heavy/data endpoints with the API key (no-op when unset). /health
# stays open for liveness probes.
_protected = [Depends(require_api_key)]
app.include_router(analysis.router, dependencies=_protected)
app.include_router(embedding.router, dependencies=_protected)
app.include_router(plate_router.router, dependencies=_protected)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(
        models_loaded={
            "clip": clip_ready(settings.clip_model_name),
            "yolo": yolo_ready(settings.yolo_model_path),
            "platdetect": platdetect_ready(settings.platdetect_model_path),
            "platreader": platreader_ready(settings.platreader_model_path),
            "motortype": motortype_ready(settings.motortype_model_path),
            "color": color_ready(settings.color_model_path),
        },
        status="ok",
        version=settings.app_version,
    )
