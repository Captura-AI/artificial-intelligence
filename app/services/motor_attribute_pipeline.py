from typing import Optional

from PIL import Image

from ..models.internal import MotorAttributeResult
from .color_classifier import classify_color
from .motor_type_detector import detect_motor_types

# Re-exported so existing callers can keep importing the DTO from this module.
__all__ = ["MotorAttributeResult", "run_motor_attributes"]


def run_motor_attributes(
    image: Image.Image,
    motortype_model_path: str,
    color_model_path: str,
    confidence_threshold: float,
) -> list[MotorAttributeResult]:
    """
    Run the motorcycle attribute flow used when a plate alone is not enough.

    1. Detect motorcycle bounding boxes + body style with motortype.pt.
    2. Crop each detected motorcycle from the original image.
    3. Classify the crop's dominant color with color.pt.

    A color classification failure is non-fatal: the detection is kept with
    color fields set to None.
    """
    detections = detect_motor_types(
        image,
        model_path=motortype_model_path,
        confidence_threshold=confidence_threshold,
    )

    results: list[MotorAttributeResult] = []
    for motor_type, type_confidence, bbox in detections:
        color: Optional[str] = None
        color_confidence: Optional[float] = None
        try:
            crop = image.crop(tuple(bbox))
            prediction = classify_color(crop, model_path=color_model_path)
            if prediction:
                color, color_confidence = prediction
        except Exception:
            pass

        results.append(
            MotorAttributeResult(
                motor_type=motor_type,
                motor_type_confidence=type_confidence,
                color=color,
                color_confidence=color_confidence,
                bbox=bbox,
            )
        )

    return results
