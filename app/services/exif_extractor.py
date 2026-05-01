import io
import struct
from typing import Optional

import piexif
import requests
from PIL import Image

from ..models.schemas import ExifData


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert GPS DMS (degrees, minutes, seconds) tuples to decimal degrees."""
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60
    seconds = dms[2][0] / dms[2][1] / 3600
    decimal = degrees + minutes + seconds
    if ref in ("S", "W"):
        decimal = -decimal
    return round(decimal, 8)


def extract_exif(image_bytes: bytes) -> ExifData:
    """Parse EXIF metadata from raw image bytes. Returns an ExifData with available fields."""
    result = ExifData()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        raw_exif = img.info.get("exif")

        if not raw_exif:
            return result

        exif_dict = piexif.load(raw_exif)

        # --- GPS ---
        gps = exif_dict.get("GPS", {})
        if gps:
            lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode("utf-8")
            lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode("utf-8")

            if lat_dms and lon_dms:
                result.latitude = _dms_to_decimal(lat_dms, lat_ref)
                result.longitude = _dms_to_decimal(lon_dms, lon_ref)

        # --- Timestamp ---
        exif_ifd = exif_dict.get("Exif", {})
        datetime_original = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
        if datetime_original:
            from datetime import datetime
            dt_str = datetime_original.decode("utf-8", errors="ignore")
            try:
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                result.captured_at = int(dt.timestamp())
            except ValueError:
                pass

        # --- Camera ---
        zeroth = exif_dict.get("0th", {})
        make = zeroth.get(piexif.ImageIFD.Make)
        model = zeroth.get(piexif.ImageIFD.Model)

        if make:
            result.camera_make = make.decode("utf-8", errors="ignore").strip().rstrip("\x00")
        if model:
            result.camera_model = model.decode("utf-8", errors="ignore").strip().rstrip("\x00")

    except Exception:
        pass

    return result


def load_image_from_url(url: str, timeout: int = 15) -> tuple[bytes, Image.Image]:
    """Download an image from a URL and return raw bytes + PIL Image."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image_bytes, image
