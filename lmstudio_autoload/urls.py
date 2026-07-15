from __future__ import annotations

from urllib.parse import urlparse


def normalize_lm_studio_base_url(url: str, *, default: str = "http://127.0.0.1:1234/v1") -> str:
    """Ensure OpenAI-compatible LM Studio base URL ends with /v1."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return default
    if u.endswith("/v1"):
        return u
    parsed = urlparse(u)
    if parsed.port == 1234 or u.endswith(":1234"):
        return f"{u}/v1"
    return u
