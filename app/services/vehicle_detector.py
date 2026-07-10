from typing import Optional

from PIL import Image

from ..config import get_settings
from .model_cache import is_model_ready as _cache_is_model_ready
from .model_cache import load_cached_model

# Maps COCO class IDs to Captura vehicle type enum values
_COCO_TO_VEHICLE_TYPE: dict[int, str] = {
    1: "BICYCLE",
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK",
}


def _load_model(model_path: str):
    """Load a YOLOv8 model from ``model_path``."""
    from ultralytics import YOLO

    return YOLO(model_path)


def _get_model(model_path: str):
    """Lazy-load YOLOv8 model on first use (cached per path)."""
    return load_cached_model(model_path, _load_model)


def detect_vehicle(
    image: Image.Image,
    model_path: str = "yolov8n.pt",
    confidence_threshold: Optional[float] = None,
    vehicle_class_ids: Optional[list[int]] = None,
) -> list[tuple[str, float, list[int]]]:
    """
    Detect vehicles in the image using YOLOv8.

    Returns:
        A list of tuples: (vehicle_type, confidence, bbox)
        where vehicle_type is one of Captura's VehicleTypeEnum values,
        and bbox is [x1, y1, x2, y2].
    """
    if confidence_threshold is None:
        confidence_threshold = get_settings().vehicle_confidence_threshold
    if vehicle_class_ids is None:
        vehicle_class_ids = list(_COCO_TO_VEHICLE_TYPE.keys())

    model = _get_model(model_path)
    results = model(image, verbose=False)

    detected_vehicles = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id not in vehicle_class_ids:
                continue
            if conf < confidence_threshold:
                continue

            bbox = [int(v) for v in box.xyxy[0].tolist()]
            vehicle_type = _COCO_TO_VEHICLE_TYPE.get(cls_id, "OTHER")
            detected_vehicles.append((vehicle_type, round(conf, 4), bbox))

    # Sort by confidence descending
    detected_vehicles.sort(key=lambda x: x[1], reverse=True)
    return detected_vehicles


def is_model_ready(model_path: str = "yolov8n.pt") -> bool:
    """Check whether the YOLO model can be loaded without errors."""
    return _cache_is_model_ready(model_path, _load_model)
