from typing import Optional

import torch
from PIL import Image

from ..config import get_settings
from .model_cache import is_model_ready as _cache_is_model_ready
from .model_cache import load_cached_model

# Tags used for zero-shot scene classification
_SCENE_TAGS = [
    "street race",
    "running marathon",
    "cycling race",
    "motorbike rider",
    "car rally",
    "off-road vehicle",
    "sports event",
    "outdoor activity",
    "urban street",
    "crowd of people",
]


def _load_model(model_name: str) -> tuple:
    """Load CLIP model + preprocess transform on the best available device.

    Returns a ``(model, preprocess, device)`` tuple. Downloads the checkpoint on
    first run (~350 MB).
    """
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device


def _get_model(model_name: str = "ViT-B/32") -> tuple:
    """Lazy-load CLIP model on first use (cached per model name)."""
    return load_cached_model(model_name, _load_model)


def get_image_embedding(
    image: Image.Image,
    model_name: str = "ViT-B/32",
) -> Optional[list[float]]:
    """
    Generate a 512-dimensional CLIP embedding for the given image.
    Returns a plain Python list of floats for JSON serialisation.
    """
    model, preprocess, device = _get_model(model_name)

    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0).to(device)
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).tolist()


def get_text_embedding(
    text: str,
    model_name: str = "ViT-B/32",
) -> Optional[list[float]]:
    """
    Generate a 512-dimensional CLIP embedding for the given text.
    Returns a plain Python list of floats for Faiss search.
    """
    import clip

    model, _, device = _get_model(model_name)

    with torch.no_grad():
        text_tensor = clip.tokenize([text]).to(device)
        features = model.encode_text(text_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).tolist()


def classify_scene_tags(
    image: Image.Image,
    model_name: str = "ViT-B/32",
    tags: Optional[list[str]] = None,
    threshold: Optional[float] = None,
) -> list[str]:
    """
    Use CLIP zero-shot classification to assign scene tags to an image.
    Returns tags whose cosine similarity exceeds `threshold`.
    """
    import clip

    if tags is None:
        tags = _SCENE_TAGS
    if threshold is None:
        threshold = get_settings().clip_tag_threshold

    model, preprocess, device = _get_model(model_name)

    text_tokens = clip.tokenize(tags).to(device)

    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)

    selected = [
        tags[i] for i, score in enumerate(similarity.tolist()) if score >= threshold
    ]
    return selected


def is_model_ready(model_name: str = "ViT-B/32") -> bool:
    """Check whether the CLIP model can be loaded."""
    return _cache_is_model_ready(model_name, _load_model)
