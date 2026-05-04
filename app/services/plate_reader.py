import re
from typing import Optional

import numpy as np
from PIL import Image

_READER = None
_READER_LANGS: list[str] = []

# Indonesian plate pattern: optional prefix letter(s), space, 1-4 digits, space, 1-3 letters
# We remove the strict word boundary (\b) to be more tolerant of OCR noise (e.g., "D 4644 LDO")
_ID_PLATE_PATTERN = re.compile(r"([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})", re.IGNORECASE)


def _get_reader(languages: list[str]):
    """Lazy-load EasyOCR reader (downloads models on first run)."""
    global _READER, _READER_LANGS
    if _READER is None or languages != _READER_LANGS:
        import easyocr
        _READER = easyocr.Reader(languages, gpu=False, verbose=False)
        _READER_LANGS = languages
    return _READER


def _normalize_plate(raw: str) -> str:
    """Normalise OCR output to uppercase with single spaces."""
    cleaned = re.sub(r"[^A-Za-z0-9 ]", "", raw)
    return " ".join(cleaned.upper().split())


def read_license_plate(
    image: Image.Image,
    languages: Optional[list[str]] = None,
    confidence_threshold: float = 0.4,
) -> tuple[Optional[str], Optional[float]]:
    """
    Extract a license plate string from an image using EasyOCR.

    Strategy:
    1. Run OCR on the full image.
    2. Filter text blocks that match the Indonesian plate pattern (e.g. 'B 1234 XYZ').
    3. Return the highest-confidence match, or the highest-confidence any-text block as fallback.

    Returns:
        (plate_text, confidence) or (None, None) if nothing credible is found.
    """
    if languages is None:
        languages = ["en"]

    reader = _get_reader(languages)
    results = reader.readtext(
        np.array(image),
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
    )

    best_plate: Optional[str] = None
    best_conf: float = 0.0

    for (_bbox, text, conf) in results:
        if conf < confidence_threshold:
            continue

        normalized = _normalize_plate(text)
        if not normalized:
            continue

        match = _ID_PLATE_PATTERN.search(normalized)
        if match:
            plate_str = f"{match.group(1)} {match.group(2)} {match.group(3)}".upper()
            if conf > best_conf:
                best_conf = conf
                best_plate = plate_str

    if best_plate is None:
        # No plate pattern matched — return the longest high-conf token as a last resort
        for (_bbox, text, conf) in results:
            if conf >= confidence_threshold and len(text.strip()) >= 4:
                normalized = _normalize_plate(text)
                if normalized and conf > best_conf:
                    best_conf = conf
                    best_plate = normalized

    if best_plate is None:
        return None, None

    return best_plate, round(best_conf, 4)


def is_model_ready(languages: list[str] = ["en"]) -> bool:
    """Check whether EasyOCR reader can be loaded."""
    try:
        _get_reader(languages)
        return True
    except Exception:
        return False
