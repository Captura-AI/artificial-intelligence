from typing import Optional

from PIL import Image

_MODEL = None
_MODEL_PATH: str = ""


def _get_model(model_path: str):
    """Lazy-load color.pt YOLOv8 classification model on first use."""
    global _MODEL, _MODEL_PATH
    if _MODEL is None or model_path != _MODEL_PATH:
        from ultralytics import YOLO
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


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
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    results = model(rgb_image, verbose=False)

    if not results or results[0].probs is None:
        return None

    probs = results[0].probs
    label = model.names.get(int(probs.top1), "Unknown")
    confidence = round(float(probs.top1conf.item()), 4)
    return label, confidence


def is_model_ready(model_path: str = "color.pt") -> bool:
    """Check whether color.pt can be loaded without errors."""
    try:
        _get_model(model_path)
        return True
    except Exception:
        return False
