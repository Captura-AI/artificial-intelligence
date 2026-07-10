import uuid
from pathlib import Path

from PIL import Image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return ``image`` in RGB mode, converting only when it is not already RGB."""
    return image if image.mode == "RGB" else image.convert("RGB")


def save_jpeg(image: Image.Image, save_dir: str, id_prefix: str) -> str:
    """Save ``image`` as a JPEG under ``save_dir`` and return its file path.

    The directory is created on demand and the file is named
    ``<id_prefix>_<uuid4hex>.jpg``. The image is converted to RGB before saving so
    non-RGB inputs (e.g. RGBA crops) serialise cleanly.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(out_dir / f"{id_prefix}_{uuid.uuid4().hex}.jpg")
    image.convert("RGB").save(file_path, format="JPEG")
    return file_path
