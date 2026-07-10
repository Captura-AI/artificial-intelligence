"""Internal pipeline DTOs.

Single source of truth for the dataclasses that flow *between* AI pipeline stages
(plate ALPR, motor attributes) before ``analyzer.py`` maps them onto the external
Pydantic response models in ``schemas.py``. These are intentionally kept separate
from ``schemas.py`` because they carry non-serialisable, implementation-detail
fields (e.g. a PIL ``Image`` crop) that must never leak into the public API.
"""

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class PlateCharacterDetection:
    text: str
    confidence: float
    bbox: list[int]


@dataclass
class PlatePipelineResult:
    text: Optional[str]
    text_confidence: Optional[float]
    detection_confidence: float
    bbox: list[int]
    crop: Image.Image
    characters: list[PlateCharacterDetection]


@dataclass
class MotorAttributeResult:
    motor_type: str
    motor_type_confidence: float
    color: Optional[str]
    color_confidence: Optional[float]
    bbox: list[int]
