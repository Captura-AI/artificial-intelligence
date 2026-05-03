# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server (reload on file change)
uvicorn app.main:app --reload --port 8000

# Run production server (matches Dockerfile CMD)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Build & run via Docker
docker build -t captura-ai .
docker run -p 8000:8000 --env-file .env captura-ai

# Lint / format
ruff check app/
black app/

# Type check
mypy app/

# Tests (pytest, no test suite exists yet)
pytest
pytest tests/path/to/test_file.py::test_function_name
```

Copy `.env.example` to `.env` before first run. Models are lazy-loaded on first request (YOLO ~6 MB, EasyOCR ~40 MB, CLIP ViT-B/32 ~350 MB) and download automatically.

## Architecture

This is a **FastAPI AI microservice** for the Captura platform. It accepts a photo URL + `moment_id` and returns structured analysis JSON intended for the NestJS backend's `moments.ai_analysis` column.

### Request lifecycle

```
POST /analyze
  └─ routers/analysis.py         # thin router — validates input, injects settings
       └─ services/analyzer.py   # pipeline orchestrator
            ├─ exif_extractor.py # downloads image, parses GPS/timestamp/camera from EXIF
            ├─ vehicle_detector.py # YOLOv8 — detects dominant vehicle type from COCO classes
            ├─ plate_reader.py   # EasyOCR — extracts Indonesian plate text (regex-matched)
            └─ clip_embedder.py  # CLIP ViT-B/32 — 512-dim embedding + zero-shot scene tags
```

The pipeline in `analyzer.py` is **fault-tolerant per step**: each AI stage is wrapped in a try/except that appends to `response.error` without aborting the remaining steps. The caller always gets a partial result.

### Model loading

All three AI models (`_MODEL` module-level globals in each service file) are lazy-loaded on first call and cached for the process lifetime. `main.py` warms them up at startup via the `lifespan` context manager to prevent cold-start latency on the first real request. The `/health` endpoint probes `is_model_ready()` on each service without re-loading.

### Configuration

`app/config.py` uses `pydantic-settings` with an `@lru_cache`-wrapped `Settings` singleton. All tunable knobs (model paths, confidence thresholds, COCO class IDs, OCR language list) live there and can be overridden via environment variables or `.env`.

### Key domain details

- **Vehicle types** map COCO class IDs (1, 2, 3, 5, 7) to Captura's `VehicleTypeEnum` strings (`BICYCLE`, `CAR`, `MOTORCYCLE`, `BUS`, `TRUCK`).
- **License plate OCR** targets Indonesian format (`B 1234 XYZ`). The regex `_ID_PLATE_PATTERN` in `plate_reader.py` is the first-pass filter; a longest-token fallback runs when no pattern matches.
- **Scene tags** in `clip_embedder.py` are hard-coded in `_SCENE_TAGS`. Cosine similarity threshold is `0.2` (`_TAG_THRESHOLD`). Add or tune tags there.
- **CLIP embeddings** are L2-normalised before returning, so downstream cosine similarity can be computed as a dot product.
