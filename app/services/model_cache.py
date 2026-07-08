from typing import Any, Callable

# Process-lifetime cache of loaded models, keyed by their path / name. Each model
# type (YOLO weights, CLIP checkpoint, ...) uses a distinct key, so a single dict
# safely holds every service's model without collision.
_MODEL_CACHE: dict[str, Any] = {}


def load_cached_model(path: str, loader_fn: Callable[[str], Any]) -> Any:
    """Lazy-load and memoise a model per unique ``path``.

    The first call for a given ``path`` invokes ``loader_fn(path)`` and caches the
    result for the lifetime of the process; subsequent calls return the cached
    instance. Different paths are cached independently, so switching model files at
    runtime loads the new one without evicting the old.
    """
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = loader_fn(path)
    return _MODEL_CACHE[path]


def is_model_ready(path: str, loader_fn: Callable[[str], Any]) -> bool:
    """Return ``True`` when the model at ``path`` can be loaded without error.

    Mirrors the original per-service ``is_model_ready`` semantics: it attempts the
    (cache-aware) load and reports success, so a warmed-up model resolves from the
    cache while a broken path surfaces as ``False``.
    """
    try:
        load_cached_model(path, loader_fn)
        return True
    except Exception:
        return False
