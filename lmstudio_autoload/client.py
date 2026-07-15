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

    def is_model_loaded(self, model_id: str) -> bool:
        """True only if the model has a non-empty loaded_instances entry (native REST API)."""
        want = model_id.strip()
        if not want:
            return False
        url = f"{self._config.api_root}/api/v1/models"
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            payload = response.json()
        for model in payload.get("models") or []:
            if not isinstance(model, dict):
                continue
            instances = model.get("loaded_instances") or []
            if not instances:
                continue
            key = (model.get("key") or "").strip()
            if key == want:
                return True
            for inst in instances:
                if isinstance(inst, dict):
                    iid = (inst.get("id") or "").strip()
                    if iid == want:
                        return True
            for variant in model.get("variants") or []:
                if isinstance(variant, str) and variant.strip() == want:
                    return True
        return False

    def load_model(self, model_id: str) -> dict[str, Any]:
        url = f"{self._config.api_root}/api/v1/models/load"
        body = {"model": model_id}
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            response = client.post(url, headers=self._headers(), json=body)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"ok": True}
