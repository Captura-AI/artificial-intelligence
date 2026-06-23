import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject requests without a valid X-API-Key when an api_key is configured.

    When `settings.api_key` is None the check is skipped entirely, so local dev
    needs no key. The comparison is constant-time to avoid leaking the key via
    timing.
    """
    if not settings.api_key:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
