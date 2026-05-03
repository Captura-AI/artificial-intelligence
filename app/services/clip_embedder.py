from typing import Optional

import torch
from PIL import Image

_MODEL = None
_PREPROCESS = None
_DEVICE: str = "cpu"
_MODEL_NAME: str = ""

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

_TAG_THRESHOLD = 0.2


def _get_model(model_name: str = "ViT-B/32"):
    """Lazy-load CLIP model (downloads on first run, ~350 MB)."""
    global _MODEL, _PREPROCESS, _DEVICE, _MODEL_NAME
    if _MODEL is None or model_name != _MODEL_NAME:
        import clip
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL, _PREPROCESS = clip.load(model_name, device=_DEVICE)
        _MODEL.eval()
        _MODEL_NAME = model_name
    return _MODEL, _PREPROCESS, _DEVICE


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
    threshold: float = _TAG_THRESHOLD,
) -> list[str]:
    """
    Use CLIP zero-shot classification to assign scene tags to an image.
    Returns tags whose cosine similarity exceeds `threshold`.
    """
    import clip

    if tags is None:
        tags = _SCENE_TAGS

    model, preprocess, device = _get_model(model_name)

    text_tokens = clip.tokenize(tags).to(device)

    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)

    selected = [tags[i] for i, score in enumerate(similarity.tolist()) if score >= threshold]
    return selected


def is_model_ready(model_name: str = "ViT-B/32") -> bool:
    """Check whether the CLIP model can be loaded."""
    try:
        _get_model(model_name)
        return True
    except Exception:
        return False
