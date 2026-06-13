"""
Сохранение настроек RAG из UI в JSON и загрузка поверх env (.env / RAG_*).

Файл по умолчанию: rag_service/ui_settings.json (рядом с пакетом).
Переопределение: RAG_UI_SETTINGS_FILE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings


def default_ui_settings_path() -> Path:
    return Path(__file__).resolve().parent / "ui_settings.json"


def resolved_ui_settings_path(s: Settings) -> Path:
    if s.ui_settings_file is not None:
        return s.ui_settings_file.expanduser().resolve()
    return default_ui_settings_path()


def _path_from_json(v: Any) -> Path | None:
    if v is None:
        return None
    if isinstance(v, Path):
        return v
    s = str(v).strip()
    if not s:
        return None
    return Path(s)


def _coerce_updates(raw: dict[str, Any]) -> dict[str, Any]:
    """Привести значения из JSON к типам полей Settings."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in Settings.model_fields:
            continue
        if k == "corpus_root":
            out[k] = _path_from_json(v)
            continue
        if k in ("embedding_device", "openai_base_url"):
            if isinstance(v, str) and not v.strip():
                out[k] = None
                continue
        if k == "ui_settings_file":
            if v is None or (isinstance(v, str) and not v.strip()):
                out[k] = None
            else:
                out[k] = Path(str(v).strip()).expanduser()
            continue
        out[k] = v
    return out


def load_settings_overlay(base: Settings) -> Settings:
    path = resolved_ui_settings_path(base)
    if not path.is_file():
        return base
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return base
    if not isinstance(raw, dict):
        return base
    updates = _coerce_updates(raw)
    if not updates:
        return base
    return base.model_copy(update=updates)


def settings_to_persist_dict(s: Settings) -> dict[str, Any]:
    """Сериализация для JSON (включая секреты — файл только на доверенной машине)."""
    d = s.model_dump(mode="json")
    # Path → str уже в mode=json
    return d


def save_settings_to_file(s: Settings, *, path: Path | None = None) -> Path:
    p = path or resolved_ui_settings_path(s)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = settings_to_persist_dict(s)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


SECRET_FIELD_NAMES = frozenset(
    ("qdrant_api_key", "lm_studio_api_key", "openai_api_key", "anthropic_api_key")
)
SETTINGS_FIELD_NAMES = frozenset(Settings.model_fields.keys())


def editable_public_dict(s: Settings) -> dict[str, Any]:
    """Плоский словарь для формы: секреты не отдаём, только признаки в отдельном блоке API."""
    d = s.model_dump(mode="json")
    for k in SECRET_FIELD_NAMES:
        if k in d and d[k]:
            d[k] = ""
    return d


def secrets_present_map(s: Settings) -> dict[str, bool]:
    return {k: bool((getattr(s, k) or "").strip()) for k in SECRET_FIELD_NAMES}


def normalize_ui_put(
    patch: dict[str, Any],
    *,
    clear_secrets: list[str] | None = None,
) -> dict[str, Any]:
    """Слияние patch + сброс секретов → словарь для model_copy(update=…)."""
    bad = set(patch) - SETTINGS_FIELD_NAMES
    if bad:
        raise ValueError(f"Неизвестные поля: {', '.join(sorted(bad))}")
    clears = list(clear_secrets or [])
    bad_clear = set(clears) - SECRET_FIELD_NAMES
    if bad_clear:
        raise ValueError(f"clear_secrets: неизвестные ключи: {', '.join(sorted(bad_clear))}")
    merged = dict(patch)
    for k in clears:
        merged[k] = None
    for k in SECRET_FIELD_NAMES:
        if k in merged and merged[k] == "":
            merged[k] = None
    return _coerce_updates(merged)
