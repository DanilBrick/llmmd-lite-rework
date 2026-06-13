from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config
from .service import AutoLoadService

app = FastAPI(title="lmstudio_autoload", version="0.1-lite")
_service = AutoLoadService(load_config())


class EnsureModelRequest(BaseModel):
    role: str


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "lmstudio_autoload"}


@app.post("/ensure-model")
def ensure_model(body: EnsureModelRequest) -> dict[str, object]:
    result = _service.ensure_model(body.role)
    if not result.ok:
        raise HTTPException(400, result.error or "ensure_model failed")
    return {
        "ok": True,
        "role": result.role,
        "model_id": result.model_id,
        "status": result.status.value,
        "already_loaded": result.already_loaded,
    }
