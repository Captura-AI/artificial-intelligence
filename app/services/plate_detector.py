from PIL import Image

from .model_cache import is_model_ready as _cache_is_model_ready
from .model_cache import load_cached_model
from .utils import ensure_rgb


def _load_model(model_path: str):
    """Load the platdetect.pt YOLOv8 model from ``model_path``."""
    from ultralytics import YOLO

    return YOLO(model_path)


def _get_model(model_path: str):
    """Lazy-load platdetect.pt YOLOv8 model on first use (cached per path)."""
    return load_cached_model(model_path, _load_model)


def detect_plates(
    image: Image.Image,
    model_path: str = "platdetect.pt",
    *,
    confidence_threshold: float,
) -> list[tuple[float, list[int]]]:
    """
    Detect license plate bounding boxes in an image using platdetect.pt.

    Returns:
        List of (confidence, bbox) tuples sorted by confidence descending,
        where bbox is [x1, y1, x2, y2] in pixel coordinates.
    """
    model = _get_model(model_path)
    rgb_image = ensure_rgb(image)
    results = model(rgb_image, conf=confidence_threshold, verbose=False)

    plates: list[tuple[float, list[int]]] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf < confidence_threshold:
                continue
            bbox = [int(v) for v in box.xyxy[0].tolist()]
            plates.append((round(conf, 4), bbox))

    plates.sort(key=lambda x: x[0], reverse=True)
    return plates


def is_model_ready(model_path: str = "platdetect.pt") -> bool:
    """Check whether platdetect.pt can be loaded without errors."""
    return _cache_is_model_ready(model_path, _load_model)
