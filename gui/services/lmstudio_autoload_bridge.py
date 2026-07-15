"""Тонкий мост из OCR job к lite lmstudio_autoload."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve_llm_models(args: dict[str, Any]) -> tuple[str, str, str, str]:
    text = (args.get("llm_text_model_id") or args.get("model_name") or "gpt-4o").strip()
    ocr_explicit = (args.get("llm_ocr_model_id") or args.get("ocr_model_name") or "").strip()
    fig = (args.get("llm_figure_model_id") or "").strip() or text
    ocr_auto = ocr_explicit or text
    return text, ocr_explicit, ocr_auto, fig


def _config_from_args(args: dict[str, Any]):
    from lmstudio_autoload.config import AutoloadConfig
    from lmstudio_autoload.urls import normalize_lm_studio_base_url

    text, _ocr_ex, ocr_auto, fig = resolve_llm_models(args)
    base = normalize_lm_studio_base_url(args.get("base_url") or "")
    key = (args.get("api_key") or "").strip() or None
    return AutoloadConfig.model_validate(
        {
            "server": {"base_url": base, "api_token": key},
            "models": {
                "ocr_model": {"model_id": ocr_auto},
                "text_model": {"model_id": text},
                "figure_model": {"model_id": fig},
            },
        }
    )


def ensure_lmstudio_roles(
    args: dict[str, Any],
    roles: list[str],
    log: Callable[[str], None],
) -> None:
    if not args.get("use_llm") or not args.get("use_lmstudio_autoload"):
        return
    url = (args.get("lmstudio_autoload_url") or "").strip().rstrip("/")
    if url:
        import httpx

        for role in roles:
            log(f"LM Studio autoload → POST {url}/ensure-model (роль «{role}»)")
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                r = client.post(f"{url}/ensure-model", json={"role": role})
                if r.status_code >= 400:
                    raise RuntimeError(f"ensure-model {role}: HTTP {r.status_code} {r.text[:500]}")
                body = r.json()
                if not body.get("ok", True):
                    raise RuntimeError(f"ensure-model {role}: {body}")
                log(f"… модель «{body.get('model_id', '?')}» готова")
        return

    from lmstudio_autoload.service import AutoLoadService

    svc = AutoLoadService(_config_from_args(args))
    for role in roles:
        log(f"LM Studio autoload: роль «{role}»")
        res = svc.ensure_model(role)
        if not res.ok:
            raise RuntimeError(res.error or f"ensure_model({role}) failed")
        log(f"… готово: {res.model_id} ({res.status.value}), уже в памяти={res.already_loaded}")
