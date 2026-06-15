from PIL import Image

_MODEL = None
_MODEL_PATH: str = ""


def _get_model(model_path: str):
    """Lazy-load motortype.pt YOLOv8 model on first use."""
    global _MODEL, _MODEL_PATH
    if _MODEL is None or model_path != _MODEL_PATH:
        from ultralytics import YOLO
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


def detect_motor_types(
    image: Image.Image,
    model_path: str = "motortype.pt",
    confidence_threshold: float = 0.4,
) -> list[tuple[str, float, list[int]]]:
    """
    Detect motorcycles and classify their body style using motortype.pt.

    Returns:
        List of (motor_type, confidence, bbox) tuples sorted by confidence
        descending, where bbox is [x1, y1, x2, y2] in pixel coordinates and
        motor_type is one of the model classes (Cruiser, Dual-Sport,
        Motocross, Scooter, Sport, Standard, Touring).
    """
    model = _get_model(model_path)
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    results = model(rgb_image, conf=confidence_threshold, verbose=False)

    detections: list[tuple[str, float, list[int]]] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0].item())
            bbox = [int(v) for v in box.xyxy[0].tolist()]
            motor_type = model.names.get(cls_id, "Unknown")
            detections.append((motor_type, round(conf, 4), bbox))

    detections.sort(key=lambda x: x[1], reverse=True)
    return detections


def is_model_ready(model_path: str = "motortype.pt") -> bool:
    """Check whether motortype.pt can be loaded without errors."""
    try:
        _get_model(model_path)
        return True
    except Exception:
        return False
