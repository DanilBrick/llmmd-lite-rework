from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .client import LMStudioClient
from .config import AutoloadConfig


class ModelStatus(str, Enum):
    ready = "ready"
    loaded = "loaded"


@dataclass(frozen=True)
class EnsureResult:
    ok: bool
    role: str
    model_id: str
    status: ModelStatus
    already_loaded: bool
    error: str | None = None


class AutoLoadService:
    def __init__(self, config: AutoloadConfig) -> None:
        self._config = config
        self._client = LMStudioClient(config)

    def ensure_model(self, role: str) -> EnsureResult:
        try:
            model_id = self._config.model_id_for_role(role)
        except (KeyError, ValueError) as e:
            return EnsureResult(
                ok=False,
                role=role,
                model_id="",
                status=ModelStatus.ready,
                already_loaded=False,
                error=str(e),
            )
        try:
            loaded = self._client.list_model_ids()
            if model_id in loaded:
                return EnsureResult(
                    ok=True,
                    role=role,
                    model_id=model_id,
                    status=ModelStatus.loaded,
                    already_loaded=True,
                )
            self._client.load_model(model_id)
            return EnsureResult(
                ok=True,
                role=role,
                model_id=model_id,
                status=ModelStatus.loaded,
                already_loaded=False,
            )
        except Exception as e:
            return EnsureResult(
                ok=False,
                role=role,
                model_id=model_id,
                status=ModelStatus.ready,
                already_loaded=False,
                error=f"{type(e).__name__}: {e}",
            )
