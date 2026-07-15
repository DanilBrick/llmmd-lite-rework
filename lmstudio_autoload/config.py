from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .urls import normalize_lm_studio_base_url


class ServerConfig(BaseModel):
    base_url: str = "http://127.0.0.1:1234/v1"
    api_token: str | None = None


class RoleModelConfig(BaseModel):
    model_id: str = ""


class AutoloadConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: dict[str, RoleModelConfig] = Field(default_factory=dict)

    def model_id_for_role(self, role: str) -> str:
        entry = self.models.get(role)
        if entry is None:
            raise KeyError(f"Unknown role: {role}")
        model_id = (entry.model_id or "").strip()
        if not model_id:
            raise ValueError(f"Model id for role '{role}' is empty in lmstudio_autoload/config.yaml")
        return model_id

    @property
    def api_root(self) -> str:
        return normalize_lm_studio_base_url(self.server.base_url).removesuffix("/v1")


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path | None = None) -> AutoloadConfig:
    cfg_path = path or default_config_path()
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    models_raw = raw.get("models") or {}
    models = {
        str(role): RoleModelConfig.model_validate(spec if isinstance(spec, dict) else {"model_id": spec})
        for role, spec in models_raw.items()
    }
    return AutoloadConfig(
        server=ServerConfig.model_validate(raw.get("server") or {}),
        models=models,
    )
