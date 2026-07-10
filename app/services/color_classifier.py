from typing import Optional

from PIL import Image

from .model_cache import is_model_ready as _cache_is_model_ready
from .model_cache import load_cached_model
from .utils import ensure_rgb


def _load_model(model_path: str):
    """Load the color.pt YOLOv8 classification model from ``model_path``."""
    from ultralytics import YOLO

    return YOLO(model_path)


def _get_model(model_path: str):
    """Lazy-load color.pt YOLOv8 classification model on first use (cached per path)."""
    return load_cached_model(model_path, _load_model)


def classify_color(
    image: Image.Image,
    model_path: str = "color.pt",
) -> Optional[tuple[str, float]]:
    """
    Classify the dominant vehicle color of an image crop using color.pt.

    Returns:
        (color_label, confidence) for the top-1 class, or None when the
        model produces no prediction. Labels are Indonesian color names
        (Biru, Emas, Hijau, Hitam, Kuning, Merah, Perak, Putih).
    """
    model = _get_model(model_path)
    rgb_image = ensure_rgb(image)
    results = model(rgb_image, verbose=False)

    if not results or results[0].probs is None:
        return None

    probs = results[0].probs
    label = model.names.get(int(probs.top1), "Unknown")
    confidence = round(float(probs.top1conf.item()), 4)
    return label, confidence


def is_model_ready(model_path: str = "color.pt") -> bool:
    """Check whether color.pt can be loaded without errors."""
    return _cache_is_model_ready(model_path, _load_model)
