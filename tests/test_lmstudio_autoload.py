from __future__ import annotations

from lmstudio_autoload.config import AutoloadConfig
from lmstudio_autoload.urls import normalize_lm_studio_base_url
from lmstudio_autoload.service import AutoLoadService


class _FakeClient:
    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.load_calls: list[str] = []

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self.loaded

    def load_model(self, model_id: str) -> dict:
        self.load_calls.append(model_id)
        self.loaded.add(model_id)
        return {"ok": True}


def test_normalize_lm_studio_base_url():
    assert normalize_lm_studio_base_url("") == "http://127.0.0.1:1234/v1"
    assert normalize_lm_studio_base_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"
    assert normalize_lm_studio_base_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234/v1"


def test_ensure_model_skips_when_already_loaded():
    cfg = AutoloadConfig.model_validate(
        {
            "server": {"base_url": "http://127.0.0.1:1234/v1"},
            "models": {"text_model": {"model_id": "qwen/text"}},
        }
    )
    svc = AutoLoadService(cfg)
    fake = _FakeClient()
    fake.loaded.add("qwen/text")
    svc._client = fake  # type: ignore[method-assign]

    result = svc.ensure_model("text_model")

    assert result.ok is True
    assert result.already_loaded is True
    assert fake.load_calls == []


def test_ensure_model_loads_missing_model():
    cfg = AutoloadConfig.model_validate(
        {
            "server": {"base_url": "http://127.0.0.1:1234/v1"},
            "models": {"ocr_model": {"model_id": "vision/model"}},
        }
    )
    svc = AutoLoadService(cfg)
    fake = _FakeClient()
    svc._client = fake  # type: ignore[method-assign]

    result = svc.ensure_model("ocr_model")

    assert result.ok is True
    assert result.already_loaded is False
    assert fake.load_calls == ["vision/model"]
