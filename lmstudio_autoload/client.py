from __future__ import annotations

from typing import Any

import httpx

from .config import AutoloadConfig


class LMStudioClient:
    def __init__(self, config: AutoloadConfig) -> None:
        self._config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = (self._config.server.api_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def list_model_ids(self) -> set[str]:
        url = f"{self._config.api_root}/v1/models"
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            payload = response.json()
        ids: set[str] = set()
        for item in payload.get("data") or []:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.add(mid.strip())
        return ids

    def load_model(self, model_id: str) -> dict[str, Any]:
        url = f"{self._config.api_root}/api/v1/models/load"
        body = {"model": model_id}
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            response = client.post(url, headers=self._headers(), json=body)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"ok": True}
