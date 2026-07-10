# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Required Reading (start of every new conversation)

Before doing any work in this service, read these two documents to ground yourself in product context:

1. `../PRD.md` — the overall Captura AI Product Requirements Document (product goals, user journeys, cross-service scope).
2. `./PRD.md` — the AI Service PRD (image analysis, OCR, vehicle detection, CLIP embedding, semantic search requirements specific to this service).

Use them to check that any change you make here still serves the product requirements, and flag if code and PRD have drifted apart.

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
  └─ routers/analysis.py           # thin router — validates input, injects settings
       └─ services/analyzer.py     # pipeline orchestrator
            ├─ exif_extractor.py   # downloads image, parses GPS/timestamp/camera from EXIF
            ├─ vehicle_detector.py # YOLOv8 — detects dominant vehicle type from COCO classes
            ├─ alpr_pipeline.py    # two-stage ALPR: plate_detector.py (YOLO plate bbox) -> plate_text_reader.py (character OCR)
            ├─ motor_attribute_pipeline.py # motor_type_detector.py + color_classifier.py
            ├─ geometry.py         # shared bbox math (IoU, plate-center-in-vehicle containment)
            └─ clip_embedder.py    # CLIP ViT-B/32 — 512-dim embedding + zero-shot scene tags
```

The pipeline in `analyzer.py` is **fault-tolerant per step**: each AI stage is wrapped in a try/except that appends to `response.error` without aborting the remaining steps. The caller always gets a partial result.

`app/services/plate_reader.py` (single-stage EasyOCR reader) was removed in the `refactor/consolidate-model-cache-geometry-helpers` cleanup — it was dead code, superseded by the `plate_detector.py` + `plate_text_reader.py` + `alpr_pipeline.py` two-stage YOLO ALPR flow above.

### Model loading

Every detector (`vehicle_detector.py`, `plate_detector.py`, `motor_type_detector.py`, `color_classifier.py`, `plate_text_reader.py`, `clip_embedder.py`) lazy-loads its model through the shared `load_cached_model()`/`is_model_ready()` in `app/services/model_cache.py` — don't hand-roll a new `_MODEL` global + loader in a new detector file, call the shared cache instead. `main.py` warms all of them up at startup via the `lifespan` context manager to prevent cold-start latency on the first real request. The `/health` endpoint probes `is_model_ready()` on each service without re-loading.

### Configuration

`app/config.py` uses `pydantic-settings` with an `@lru_cache`-wrapped `Settings` singleton. All tunable knobs (model paths, confidence thresholds, COCO class IDs, OCR language list) live there and can be overridden via environment variables or `.env`.

### Key domain details

- **Vehicle types** map COCO class IDs (1, 2, 3, 5, 7) to Captura's `VehicleTypeEnum` strings (`BICYCLE`, `CAR`, `MOTORCYCLE`, `BUS`, `TRUCK`).
- **License plate OCR** targets Indonesian format (`B 1234 XYZ`) via the two-stage `alpr_pipeline.py` (plate bbox detection, then per-character reading in `plate_text_reader.py`).
- **Scene tags** in `clip_embedder.py` are hard-coded in `_SCENE_TAGS`. Cosine similarity threshold is `clip_tag_threshold` in `config.py`'s `Settings` (default `0.2`) — not a bare module constant, so tune it via env/config rather than editing the file.
- **CLIP embeddings** are L2-normalised before returning, so downstream cosine similarity can be computed as a dot product.

### Clean Code Standard (MANDATORY)

All code must be clean, optimized, and consistent with current best practices. Primary references:

1. Clean Code (general) — https://github.com/ryanmcdermott/clean-code-javascript (principles translate directly even though the repo is JS-flavored)
2. PEP 8 — https://peps.python.org/pep-0008/
3. FastAPI docs — https://fastapi.tiangolo.com

Non-negotiable rules:

- **Vertical spacing for readability:** blank line before `if`/`for`/`try`/`return` blocks when separating them from the statement above.
- **Guard clauses / early return:** avoid nesting beyond 2–3 levels.
- **Descriptive names:** no ambiguous abbreviations; boolean-returning functions start with `is_`/`has_`.
- **Small, single-responsibility functions:** target < 50 lines; split when larger.
- **Type hints everywhere:** function signatures fully typed; use `Optional[...]`/`| None` explicitly, never an untyped `dict`/`Any` when a Pydantic model or dataclass would do.
- **Never fight the formatter:** `ruff check app/` and `black app/` must be clean before any PR.
- **Match existing patterns** (routers thin, services do the work, shared logic in dedicated modules) rather than introducing new styles.

#### No Duplicated Constants/Helpers/Types Across Services (MANDATORY)

A `services/<name>.py` file holds that model/pipeline's own logic only — never a hand-rolled copy of something that already exists (or should exist) in a shared module. This was a real problem (see `refactor/consolidate-model-cache-geometry-helpers`): six files independently reimplemented the same lazy-load-singleton pattern, three files reimplemented the same bbox-containment math, and confidence thresholds were duplicated between `config.py` and hardcoded function defaults.

Where things go:

1. **Model loading** → `app/services/model_cache.py` (`load_cached_model`, `is_model_ready`). Never write a new `_MODEL`/`_get_model()` pair by hand.
2. **Image helpers used by 2+ files** (RGB conversion, saving crops to disk) → `app/services/utils.py` (`ensure_rgb`, `save_jpeg`).
3. **bbox/geometry math used by 2+ files** → `app/services/geometry.py` (`_bbox_iou`, `_plate_center`, `_bbox_contains_point`, `_expand_bbox`). If a router needs this math (e.g. `routers/plate.py`), import it — don't re-derive it inline.
4. **Internal pipeline DTOs shared across 2+ files** (not part of the external API) → `app/models/internal.py`. Keep these separate from `app/models/schemas.py`, which is external-API-response models only — internal DTOs may carry non-serializable fields (e.g. a PIL `Image` crop) that must never leak into an API response.
5. **Tunable numeric thresholds** (confidence, padding, similarity cutoffs) → `app/config.py`'s `Settings` class only. Never re-declare the same threshold as a hardcoded function-parameter default "for safety" — if every caller already passes it explicitly, make the parameter required instead; if a caller relies on the default, default to `None` and resolve via `get_settings()` inside the function body.

Before writing a new module-level constant, dataclass, or helper function inside a service file: grep for the name/shape across `app/services/` first. If equivalent logic already exists elsewhere, import and reuse it.
