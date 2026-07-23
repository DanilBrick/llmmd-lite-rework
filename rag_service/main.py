"""
HTTP API RAG-сервиса: индексация каталога с .md (результат MarkItDown GUI), поиск в Qdrant, опционально ответ LLM.

Режимы нарезки (RAG_CHUNKING_MODE): heading — как в GUI (уровень #); semantic — LM Studio
(OpenAI-совместимый API); heading_semantic — сначала заголовки, затем LM для длинных секций.
LM Studio: RAG_LM_STUDIO_BASE_URL (по умолчанию http://127.0.0.1:1234/v1), RAG_SEMANTIC_CHUNK_MODEL.

POST /v1/rag — выбор генератора: поле llm_provider = lm_studio | openai | anthropic | auto.
  lm_studio — локальный chat/completions (RAG_LM_STUDIO_BASE_URL, модель: model в теле или RAG_LM_STUDIO_RAG_MODEL или RAG_SEMANTIC_CHUNK_MODEL).
  openai — облако / любой OpenAI-совместимый (RAG_OPENAI_BASE_URL, RAG_OPENAI_API_KEY, RAG_DEFAULT_LLM_MODEL).
  anthropic — Claude (RAG_ANTHROPIC_API_KEY, RAG_ANTHROPIC_MODEL).
  auto — RAG_DEFAULT_RAG_LLM_PROVIDER или по наличию ключей: openai → anthropic → lm_studio.

Запуск из корня репозитория:
  pip install -r rag_service/requirements-rag.txt
  docker run -p 6333:6333 qdrant/qdrant
  python llmmd.py rag

Веб-дашборд: http://127.0.0.1:8765/ · JSON: GET /v1/status · GET /v1/config (без секретов)
Векторное хранилище: GET /v1/qdrant/collection — метаданные коллекции · GET /v1/qdrant/points — срез точек (scroll)
Настройки из UI: GET /v1/settings · PUT /v1/settings (тело { "patch": {…}, "clear_secrets": [] }) → rag_service/ui_settings.json
Индексация в фоне: POST /v1/index с телом { "corpus_root": "...", "wait": false }

Переменные окружения (префикс RAG_): см. rag_service.config.Settings; JSON из UI перекрывает env при старте и после сохранения.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

_STOP_LM_MODELS = ("intfloat/multilingual-e5-large", "nomic-embed-text-v1.5-GGUF")

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .chunking import chunk_markdown_file, iter_markdown_files, find_or_convert_docx_files
from .config import Settings
from .semantic_chunking import normalize_chunking_mode, split_for_index
from .embeddings import HybridEncoder, load_hybrid_encoder
from .runtime_status import (
    activity_snapshot,
    index_job_finish_fail,
    index_job_finish_ok,
    index_job_begin,
    index_job_append_error,
    index_job_progress,
    index_job_snapshot,
    is_index_job_running,
    new_job_id,
    note_rag,
    note_search,
    service_snapshot,
    set_service_ready,
    set_service_starting,
)
from .store import (
    ensure_collection,
    recreate_collection,
    search_dense,
    search_hybrid,
    stable_point_id,
    upsert_chunks,
)
from .ui_settings_store import (
    editable_public_dict,
    load_settings_overlay,
    normalize_ui_put,
    resolved_ui_settings_path,
    save_settings_to_file,
    secrets_present_map,
)

_log = logging.getLogger(__name__)

settings = load_settings_overlay(Settings())


class AppState:
    settings: Settings
    encoder: HybridEncoder | None = None
    hybrid: bool = False
    qdrant: Any = None
    dense_dim: int = 0
    startup_error: str | None = None


state = AppState()
state.settings = settings

_settings_lock = threading.Lock()


def _dense_dim(enc: HybridEncoder) -> int:
    dm = enc.dense_model
    fn = getattr(dm, "get_embedding_dimension", None)
    if callable(fn):
        return int(fn())
    return int(dm.get_sentence_embedding_dimension())


def bootstrap_from_settings(s: Settings) -> tuple[Any, HybridEncoder | None, bool, int, str | None]:
    """Подключение Qdrant и загрузка энкодера. (qd, enc, hybrid, dense_dim, err)."""
    from qdrant_client import QdrantClient

    try:
        qd = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        qd.get_collections()
        enc, hybrid = load_hybrid_encoder(
            s.embedding_model,
            device=s.embedding_device,
            enable_hybrid=s.enable_hybrid,
        )
        dim = _dense_dim(enc)
        ensure_collection(qd, s.collection_name, dim, hybrid)
        return qd, enc, hybrid, dim, None
    except Exception as e:
        return None, None, False, 0, f"{type(e).__name__}: {e}"


def _needs_qdrant_encoder_refresh(old: Settings, new: Settings) -> bool:
    return any(
        getattr(old, k) != getattr(new, k)
        for k in ("qdrant_url", "qdrant_api_key", "collection_name", "embedding_model", "embedding_device", "enable_hybrid")
    )


def _api_bind_changed(old: Settings, new: Settings) -> bool:
    return old.api_host != new.api_host or old.api_port != new.api_port


def _close_qdrant_client(qd: Any) -> None:
    if qd is None:
        return
    close = getattr(qd, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = state.settings
    state.startup_error = None
    set_service_starting()
    
    # Проверка модели semantic_chunk_model
    semantic_model = (s.semantic_chunk_model or "").strip()
    if semantic_model in _STOP_LM_MODELS:
        state.startup_error = (
            f"semantic_chunk_model={semantic_model} — это модель эмбеддингов, а не LLM. "
            f"Укажите имя LLM-модели (например, google/gemma-3-1b) в RAG_SEMANTIC_CHUNK_MODEL"
        )
        set_service_ready(
            startup_error=state.startup_error,
            embedding_model="",
            hybrid=False,
            dense_dim=0,
            qdrant_url=s.qdrant_url,
            collection_name=s.collection_name,
        )
        yield
        return
    
    qd, enc, hybrid, dim, err = bootstrap_from_settings(s)
    if err:
        state.startup_error = err
        state.qdrant = None
        state.encoder = None
        state.dense_dim = 0
        set_service_ready(
            startup_error=state.startup_error,
            embedding_model="",
            hybrid=False,
            dense_dim=0,
            qdrant_url=s.qdrant_url,
            collection_name=s.collection_name,
        )
    else:
        state.qdrant = qd
        state.encoder = enc
        state.hybrid = hybrid
        state.dense_dim = dim
        set_service_ready(
            startup_error=None,
            embedding_model=s.embedding_model,
            hybrid=hybrid,
            dense_dim=state.dense_dim,
            qdrant_url=s.qdrant_url,
            collection_name=s.collection_name,
        )
    yield
    _close_qdrant_client(state.qdrant)
    state.qdrant = None
    state.encoder = None
    state.startup_error = None


RAG_API_VERSION = "0.2.0"

app = FastAPI(title="llmmd RAG (Qdrant)", version=RAG_API_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    ok = state.qdrant is not None and state.encoder is not None
    out: dict[str, Any] = {
        "ok": ok,
        "hybrid": state.hybrid,
        "dense_dim": state.dense_dim,
        "startup_error": state.startup_error,
    }
    if ok:
        try:
            state.qdrant.get_collection(state.settings.collection_name)
            out["qdrant"] = "connected"
        except Exception as e:
            out["qdrant"] = f"error: {e}"
    s = state.settings
    out["chunking_mode"] = s.chunking_mode
    out["lm_studio_base_url"] = s.lm_studio_base_url
    out["semantic_chunk_model_configured"] = bool((s.semantic_chunk_model or "").strip())
    out["dashboard"] = "/"
    out["api_version"] = RAG_API_VERSION
    return out


@app.get("/v1/lm-studio/health")
def lm_studio_health() -> dict[str, Any]:
    """Проверка доступности LM Studio (GET …/v1/models)."""
    s = state.settings
    root = (s.lm_studio_base_url or "").strip().rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    url = root + "/models"
    headers: dict[str, str] = {}
    if s.lm_studio_api_key:
        headers["Authorization"] = f"Bearer {s.lm_studio_api_key}"
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(url, headers=headers)
        ok = r.status_code == 200
        data = r.json() if ok else None
        n = None
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            n = len(data["data"])
        return {"ok": ok, "status_code": r.status_code, "url": url, "model_count": n}
    except Exception as e:
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}


class IndexRequest(BaseModel):
    corpus_root: Optional[Path] = Field(default=None, description="Каталог с .md/.docx; иначе RAG_CORPUS_ROOT")
    recreate_collection: bool = False
    glob_pattern: str = "**/*.docx"
    heading_level: Optional[int] = Field(default=None, ge=1, le=6, description="Переопределить RAG_HEADING_LEVEL")
    chunk_max_chars: Optional[int] = Field(
        default=None,
        ge=0,
        description="Переопределить RAG_CHUNK_MAX_CHARS (0 = только заголовки)",
    )
    chunking_mode: Optional[str] = Field(
        default=None,
        description="heading | semantic | heading_semantic; иначе RAG_CHUNKING_MODE",
    )
    wait: bool = Field(
        default=True,
        description="true — дождаться конца и вернуть результат; false — фон (смотри GET /v1/status)",
    )


def _read_dashboard_html() -> str:
    return (Path(__file__).resolve().parent / "dashboard.html").read_text(encoding="utf-8")


def _corpus_root_display(s: Settings) -> str | None:
    if s.corpus_root is None:
        return None
    try:
        return str(s.corpus_root.resolve())
    except Exception:
        return str(s.corpus_root)


@app.get("/v1/config")
def public_config() -> dict[str, Any]:
    """Настройки без секретов — для дашборда (ключи только как «задано / нет»)."""
    s = state.settings

    def key_set(v: str | None) -> bool:
        return bool((v or "").strip())

    return {
        "api_version": RAG_API_VERSION,
        "api_listen": f"{s.api_host}:{s.api_port}",
        "qdrant_url": s.qdrant_url,
        "qdrant_api_key_set": key_set(s.qdrant_api_key),
        "collection_name": s.collection_name,
        "corpus_root_default": _corpus_root_display(s),
        "embedding_model": s.embedding_model,
        "embedding_device": s.embedding_device,
        "enable_hybrid": s.enable_hybrid,
        "chunking_mode": s.chunking_mode,
        "heading_level": s.heading_level,
        "chunk_max_chars": s.chunk_max_chars,
        "chunk_overlap_chars": s.chunk_overlap_chars,
        "rag_context_max_chars": s.rag_context_max_chars,
        "rag_source_max_chars": s.rag_source_max_chars,
        "rag_dedupe_sources": s.rag_dedupe_sources,
        "lm_studio_base_url": s.lm_studio_base_url,
        "lm_studio_api_key_set": key_set(s.lm_studio_api_key),
        "semantic_chunk_model": (s.semantic_chunk_model or "").strip() or None,
        "lm_studio_rag_model": (s.lm_studio_rag_model or "").strip() or None,
        "semantic_llm_timeout_s": s.semantic_llm_timeout_s,
        "semantic_chunk_max_input_chars": s.semantic_chunk_max_input_chars,
        "semantic_subchunk_min_chars": s.semantic_subchunk_min_chars,
        "semantic_chunk_temperature": s.semantic_chunk_temperature,
        "openai_base_url": s.openai_base_url,
        "openai_api_key_set": key_set(s.openai_api_key),
        "default_llm_model": s.default_llm_model,
        "default_rag_llm_provider": s.default_rag_llm_provider,
        "anthropic_model": s.anthropic_model,
        "anthropic_api_key_set": key_set(s.anthropic_api_key),
        "anthropic_max_tokens": s.anthropic_max_tokens,
        "secrets_note": "Ключи не отдаются в GET. Задайте через env (RAG_*), через rag_service/ui_settings.json или панель: GET/PUT /v1/settings.",
    }


class UiSettingsPutBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patch: dict[str, Any] = Field(default_factory=dict, description="Поля Settings по имени")
    clear_secrets: list[str] = Field(
        default_factory=list,
        description="Имена секретных полей для сброса: qdrant_api_key, lm_studio_api_key, openai_api_key, anthropic_api_key",
    )


@app.get("/v1/settings")
def ui_settings_get() -> dict[str, Any]:
    """Текущие настройки для формы (секреты — пустые строки, см. secrets_present)."""
    s = state.settings
    path = resolved_ui_settings_path(s)
    return {
        "persist_file": str(path),
        "persist_file_exists": path.is_file(),
        "current": editable_public_dict(s),
        "secrets_present": secrets_present_map(s),
        "note": "Сохранённые секреты лежат в JSON на диске — только при локальном доступе. Не открывайте API в интернет без авторизации.",
    }


@app.put("/v1/settings")
def ui_settings_put(body: UiSettingsPutBody) -> dict[str, Any]:
    """Обновить настройки, сохранить в JSON и при необходимости переподключить Qdrant/эмбеддинг."""
    if not body.patch and not body.clear_secrets:
        p = resolved_ui_settings_path(state.settings)
        return {
            "ok": True,
            "message": "Пустой запрос — без изменений",
            "saved_to": str(p),
        }
    try:
        updates = normalize_ui_put(body.patch, clear_secrets=body.clear_secrets)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not updates:
        return {
            "ok": True,
            "message": "Нет полей для обновления",
            "saved_to": str(resolved_ui_settings_path(state.settings)),
        }

    with _settings_lock:
        old_settings = state.settings
        try:
            new_s = old_settings.model_copy(update=updates)
        except ValidationError as e:
            raise HTTPException(400, str(e)) from e
        refresh = _needs_qdrant_encoder_refresh(old_settings, new_s)
        if refresh and is_index_job_running():
            raise HTTPException(
                409,
                "Идёт индексация — дождитесь завершения или измените настройки без смены Qdrant URL/ключа или модели эмбеддинга.",
            )
        msgs: list[str] = []
        if _api_bind_changed(old_settings, new_s):
            msgs.append(
                "api_host / api_port записаны в файл; чтобы сменить адрес прослушивания процесса, перезапустите сервис вручную."
            )
        if refresh:
            qd, enc, hybrid, dim, err = bootstrap_from_settings(new_s)
            if err:
                raise HTTPException(400, f"Не удалось применить настройки: {err}") from None
            old_qd = state.qdrant
            state.qdrant = qd
            state.encoder = enc
            state.hybrid = hybrid
            state.dense_dim = dim
            state.settings = new_s
            state.startup_error = None
            set_service_ready(
                startup_error=None,
                embedding_model=new_s.embedding_model,
                hybrid=hybrid,
                dense_dim=dim,
                qdrant_url=new_s.qdrant_url,
                collection_name=new_s.collection_name,
            )
            _close_qdrant_client(old_qd)
        else:
            state.settings = new_s
        path = save_settings_to_file(state.settings)
    out: dict[str, Any] = {"ok": True, "saved_to": str(path)}
    if msgs:
        out["messages"] = msgs
    return out


def _execute_index_core(
    body: IndexRequest,
    root: Path,
    chunking_mode: str,
    job_id: str,
    files: list[Path],
    *,
    track: bool,
    wait_mode: str,
) -> dict[str, Any]:
    s = state.settings
    if state.encoder is None or state.qdrant is None:
        raise RuntimeError("Сервис не готов")
    if track:
        index_job_begin(
            job_id,
            wait_mode=wait_mode,
            corpus_root=str(root),
            glob_pattern=body.glob_pattern,
            chunking_mode=chunking_mode,
            recreate_collection=body.recreate_collection,
            files_total=len(files),
        )
    try:
        if body.recreate_collection:
            recreate_collection(state.qdrant, s.collection_name, state.dense_dim, state.hybrid)
        else:
            ensure_collection(state.qdrant, s.collection_name, state.dense_dim, state.hybrid)

        t0 = time.perf_counter()
        total_chunks = 0
        indexed_files = 0
        skipped = 0
        errors: list[str] = []

        hl = body.heading_level if body.heading_level is not None else s.heading_level
        mc = body.chunk_max_chars if body.chunk_max_chars is not None else s.chunk_max_chars

        for i, path in enumerate(files):
            rel = path.relative_to(root).as_posix()
            if track:
                index_job_progress(
                    files_done=i,
                    chunks_upserted=total_chunks,
                    current_file=rel,
                    current_stage="index_file",
                    files_skipped_empty=skipped,
                )
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                if chunking_mode == "heading":
                    chunks = chunk_markdown_file(
                        text,
                        heading_level=hl,
                        chunk_max_chars=mc,
                        chunk_overlap_chars=s.chunk_overlap_chars,
                    )
                else:
                    chunks = split_for_index(
                        text,
                        s,
                        chunking_mode,
                        heading_level=hl,
                        chunk_max_chars=mc,
                        chunk_overlap_chars=s.chunk_overlap_chars,
                    )
                if not chunks:
                    skipped += 1
                    continue
                tuples = [(c.text, c.heading, c.chunk_index) for c in chunks]
                n = upsert_chunks(
                    state.qdrant,
                    s.collection_name,
                    state.encoder,
                    state.hybrid,
                    corpus_root=root,
                    rel_path=rel,
                    chunks=tuples,
                )
                total_chunks += n
                indexed_files += 1
            except Exception as e:
                err = f"{path}: {e!s}"
                errors.append(err)
                if track:
                    index_job_append_error(err)

        dt = time.perf_counter() - t0
        summary: dict[str, Any] = {
            "corpus_root": str(root),
            "indexed_files": indexed_files,
            "chunks_upserted": total_chunks,
            "seconds": round(dt, 3),
            "hybrid": state.hybrid,
            "heading_level": hl,
            "chunk_max_chars": mc,
            "chunking_mode": chunking_mode,
            "lm_studio_base_url": s.lm_studio_base_url,
            "errors": errors[:50],
            "error_count": len(errors),
            "files_skipped_empty": skipped,
            "job_id": job_id,
            "wait": wait_mode == "sync",
        }
        if track:
            index_job_progress(
                files_done=len(files),
                chunks_upserted=total_chunks,
                current_file=None,
                current_stage="done",
                files_skipped_empty=skipped,
            )
            index_job_finish_ok(summary)
        return summary
    except Exception as e:
        if track:
            index_job_finish_fail(f"{type(e).__name__}: {e}")
        raise


def _index_job_status_dict(job: Any) -> dict[str, Any]:
    d = asdict(job)
    errs = d.pop("errors", []) or []
    d["recent_errors"] = errs[-40:]
    return d


@app.get("/", response_class=HTMLResponse)
def dashboard_ui() -> str:
    return _read_dashboard_html()


@app.get("/v1/status")
def full_status() -> dict[str, Any]:
    """Сводка для UI: сервис, Qdrant, индексация, LM Studio, активность."""
    svc_m = service_snapshot()
    job = index_job_snapshot()
    act = activity_snapshot()
    s = state.settings

    ready = state.qdrant is not None and state.encoder is not None and not svc_m.startup_error
    qd: dict[str, Any] = {
        "collection_name": s.collection_name,
        "collection_ok": False,
        "points_count": None,
        "collections_count": None,
        "error": None,
    }
    if state.qdrant is not None:
        try:
            qd["collections_count"] = len(_qdrant_collection_names())
            state.qdrant.get_collection(s.collection_name)
            qd["collection_ok"] = True
            cnt = state.qdrant.count(collection_name=s.collection_name, exact=False)
            qd["points_count"] = int(getattr(cnt, "count", cnt) or 0)
        except Exception as e:
            qd["error"] = f"{type(e).__name__}: {e}"

    lm = lm_studio_health()

    now = time.time()
    uptime_ready = (now - svc_m.ready_at_unix) if svc_m.ready_at_unix else None
    uptime_started = now - svc_m.started_at_unix if svc_m.started_at_unix else None

    return {
        "meta": {
            "api_version": RAG_API_VERSION,
            "uptime_since_start_s": round(uptime_started, 1) if uptime_started is not None else None,
            "uptime_since_ready_s": round(uptime_ready, 1) if uptime_ready is not None else None,
        },
        "service": {
            "ready": ready,
            "phase": svc_m.phase,
            "startup_error": svc_m.startup_error,
            "embedding_model": svc_m.embedding_model,
            "hybrid_enabled": svc_m.hybrid_enabled,
            "dense_dim": svc_m.dense_dim,
            "qdrant_url": svc_m.qdrant_url,
            "started_at_unix": svc_m.started_at_unix,
            "ready_at_unix": svc_m.ready_at_unix,
            "chunking_mode_default": s.chunking_mode,
            "semantic_chunk_model_configured": bool((s.semantic_chunk_model or "").strip()),
            "rag_context_max_chars": s.rag_context_max_chars,
            "rag_source_max_chars": s.rag_source_max_chars,
        },
        "qdrant": qd,
        "index_job": _index_job_status_dict(job),
        "activity": act.__dict__,
        "lm_studio": {
            "base_url": s.lm_studio_base_url,
            "model_configured": bool((s.semantic_chunk_model or "").strip()),
            "server_ok": lm.get("ok"),
            "model_count": lm.get("model_count"),
            "error": lm.get("error"),
        },
        "rag_llm": {
            "default_rag_llm_provider": s.default_rag_llm_provider,
            "lm_studio_rag_ready": bool(
                (s.lm_studio_base_url or "").strip()
                and (
                    (s.lm_studio_rag_model or "").strip()
                    or (s.semantic_chunk_model or "").strip()
                ),
            ),
            "openai_ready": bool((s.openai_api_key or "").strip() or (s.openai_base_url or "").strip()),
            "anthropic_ready": bool((s.anthropic_api_key or "").strip()),
        },
    }


@app.post("/v1/index")
def index_corpus(body: IndexRequest) -> Any:
    if state.encoder is None or state.qdrant is None:
        raise HTTPException(
            503,
            f"Сервис не готов. {state.startup_error or 'Проверьте Qdrant (docker compose -f rag_service/docker-compose.qdrant.yml up -d).'}",
        )
    root = body.corpus_root or state.settings.corpus_root
    if root is None:
        raise HTTPException(
            400,
            "Укажите corpus_root в теле запроса или задайте переменную RAG_CORPUS_ROOT",
        )
    root = root.resolve()
    if not root.is_dir():
        raise HTTPException(400, f"Каталог не найден: {root}")

    s = state.settings
    mode_raw = body.chunking_mode if body.chunking_mode is not None else s.chunking_mode
    try:
        chunking_mode = normalize_chunking_mode(mode_raw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    files = list(find_or_convert_docx_files(root, glob_pattern=body.glob_pattern))
    job_id = new_job_id()

    if not body.wait:
        if is_index_job_running():
            raise HTTPException(409, "Индексация уже выполняется — дождитесь завершения (GET /v1/status)")

        def runner() -> None:
            try:
                _execute_index_core(
                    body,
                    root,
                    chunking_mode,
                    job_id,
                    files,
                    track=True,
                    wait_mode="async",
                )
            except Exception as e:
                index_job_finish_fail(f"{type(e).__name__}: {e}")

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse(
            {
                "started": True,
                "job_id": job_id,
                "wait": False,
                "files_total": len(files),
                "poll": "/v1/status",
                "ui": "/",
            },
            status_code=202,
        )

    try:
        return _execute_index_core(
            body,
            root,
            chunking_mode,
            job_id,
            files,
            track=True,
            wait_mode="sync",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=8, ge=1, le=64)
    score_threshold: Optional[float] = None


def run_search(body: SearchRequest) -> dict[str, Any]:
    if state.encoder is None or state.qdrant is None:
        raise HTTPException(503, state.startup_error or "Сервис не готов")
    q = body.query.strip()
    if not q:
        raise HTTPException(400, "Пустой query")

    dense_q = state.encoder.encode_query(q)
    hits: list[Any]

    if state.hybrid and state.encoder.sparse_model is not None:
        sparse_vecs = state.encoder.encode_sparse([q])
        if sparse_vecs and len(sparse_vecs) == 1:
            hits = search_hybrid(
                state.qdrant,
                state.settings.collection_name,
                dense_q,
                sparse_vecs[0],
                body.limit,
            )
        else:
            hits = search_dense(
                state.qdrant,
                state.settings.collection_name,
                dense_q,
                body.limit,
                body.score_threshold,
                named_dense=True,
            )
    else:
        hits = search_dense(
            state.qdrant,
            state.settings.collection_name,
            dense_q,
            body.limit,
            body.score_threshold,
            named_dense=False,
        )

    items: list[dict[str, Any]] = []
    for p in hits:
        pl = p.payload or {}
        items.append(
            {
                "id": str(p.id),
                "score": float(p.score) if p.score is not None else None,
                "source_path": pl.get("source_path"),
                "heading": pl.get("heading"),
                "text": pl.get("text"),
            }
        )
    note_search(q)
    return {"query": body.query, "hits": items}


@app.post("/v1/search")
def search_endpoint(body: SearchRequest) -> dict[str, Any]:
    return run_search(body)


SYSTEM_RU = """Ты помощник по технической и нормативной литературе. Отвечай только на основе переданных фрагментов источников.
Если в источниках нет ответа, так и скажи. Ссылайся на номер источника в квадратных скобках, например [1]."""


class RagRequest(BaseModel):
    query: str
    topic: Optional[str] = None
    limit: int = Field(default=6, ge=1, le=32)
    max_context_chars: Optional[int] = Field(
        default=None,
        ge=1000,
        le=200000,
        description="Переопределить RAG_CONTEXT_MAX_CHARS для одного запроса",
    )
    max_source_chars: Optional[int] = Field(
        default=None,
        ge=500,
        le=100000,
        description="Переопределить RAG_SOURCE_MAX_CHARS для одного источника",
    )
    model: Optional[str] = None
    llm_provider: Optional[str] = Field(
        default=None,
        description="lm_studio | openai | anthropic | auto (или пусто = auto)",
    )


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "llmmd-rag"
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    llm_provider: Optional[str] = Field(
        default=None,
        description="Optional llmmd extension: lm_studio | openai | anthropic | auto.",
    )
    limit: int = Field(default=6, ge=1, le=32)


def _openai_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in (None, "text", "input_text"):
                    value = item.get("text") or item.get("content")
                    if isinstance(value, str):
                        parts.append(value)
        return "\n".join(p for p in parts if p.strip())
    return str(content)


def _last_openai_user_text(messages: list[OpenAIChatMessage]) -> str:
    for message in reversed(messages):
        if (message.role or "").lower() == "user":
            text = _openai_content_to_text(message.content).strip()
            if text:
                return text
    for message in reversed(messages):
        text = _openai_content_to_text(message.content).strip()
        if text:
            return text
    return ""


def _model_for_provider(requested_model: str | None) -> str | None:
    model = (requested_model or "").strip()
    if not model or model in {"llmmd-rag", "llmmd-search"}:
        return None
    return model


def _openai_sources_text(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = ["", "", "Источники:"]
    for i, src in enumerate(sources, start=1):
        path = src.get("source_path") or ""
        heading = src.get("heading") or ""
        label = f"[{i}] {path}".strip()
        if heading:
            label += f" - {heading}"
        lines.append(label)
    return "\n".join(lines)


def _chat_completion_payload(
    *,
    request_model: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    created = int(time.time())
    content = answer + _openai_sources_text(sources)
    return {
        "id": f"chatcmpl-llmmd-{created}",
        "object": "chat.completion",
        "created": created,
        "model": request_model or "llmmd-rag",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "llmmd": {
            "sources": sources,
        },
    }


def _sse_line(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.get("/v1/models")
def openai_models() -> dict[str, Any]:
    """OpenAI-compatible model list for Open WebUI."""
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "llmmd-rag",
                "object": "model",
                "created": created,
                "owned_by": "llmmd",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: OpenAIChatCompletionRequest) -> Any:
    """OpenAI-compatible facade for Open WebUI: last user message -> /v1/rag."""
    query = _last_openai_user_text(body.messages)
    if not query:
        raise HTTPException(400, "No user message content found")

    rag_payload = await rag(
        RagRequest(
            query=query,
            topic=query,
            limit=body.limit,
            model=_model_for_provider(body.model),
            llm_provider=body.llm_provider,
        )
    )
    answer = str(rag_payload.get("answer") or "")
    sources = list(rag_payload.get("sources") or [])
    payload = _chat_completion_payload(
        request_model=body.model,
        answer=answer,
        sources=sources,
    )
    if not body.stream:
        return payload

    async def _events():
        created = payload["created"]
        model = payload["model"]
        content = payload["choices"][0]["message"]["content"]
        yield _sse_line(
            {
                "id": payload["id"],
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
            }
        )
        yield _sse_line(
            {
                "id": payload["id"],
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


def _normalize_rag_provider(request_value: str | None, s: Settings) -> str:
    raw = (request_value or "").strip().lower()
    if raw in ("", "auto", "automatic"):
        d = (s.default_rag_llm_provider or "auto").strip().lower()
        if d not in ("", "auto", "automatic"):
            raw = d
    if raw in ("", "auto", "automatic"):
        if (s.openai_api_key or "").strip() or (s.openai_base_url or "").strip():
            return "openai"
        if (s.anthropic_api_key or "").strip():
            return "anthropic"
        return "lm_studio"
    if raw in ("lm_studio", "lm-studio", "local"):
        return "lm_studio"
    if raw in ("openai", "oai", "open_ai"):
        return "openai"
    if raw in ("anthropic", "claude"):
        return "anthropic"
    raise ValueError(f"Неизвестный llm_provider: {request_value!r}. Допустимо: auto, lm_studio, openai, anthropic.")


def _lm_studio_model_for_rag(body: RagRequest, s: Settings) -> str:
    return (
        (body.model or "").strip()
        or (s.lm_studio_rag_model or "").strip()
        or (s.semantic_chunk_model or "").strip()
    )


def _rag_context_from_hits(
    hits: list[dict[str, Any]],
    *,
    max_context_chars: int,
    max_source_chars: int,
    dedupe_sources: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Build bounded RAG context so one large collection cannot overflow the LLM prompt."""
    blocks: list[str] = []
    used_hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    skipped_duplicates = 0
    skipped_budget = 0
    truncated_sources = 0
    total_original_chars = 0
    total_used_chars = 0

    for h in hits:
        src = str(h.get("source_path") or "")
        head = str(h.get("heading") or "")
        txt = str(h.get("text") or "").strip()
        if not txt:
            continue
        total_original_chars += len(txt)
        if dedupe_sources:
            key = (src, head, txt[:500])
            if key in seen:
                skipped_duplicates += 1
                continue
            seen.add(key)

        source_truncated = len(txt) > max_source_chars
        if source_truncated:
            txt = txt[:max_source_chars].rstrip() + "\n\n[... источник обрезан по лимиту RAG_SOURCE_MAX_CHARS ...]"
            truncated_sources += 1

        source_no = len(used_hits) + 1
        block = f"[ИСТОЧНИК {source_no}] (файл: {src}; раздел: {head})\n{txt}"
        sep_len = 2 if blocks else 0
        remaining = max_context_chars - total_used_chars - sep_len
        if remaining <= 0:
            skipped_budget += 1
            continue
        if len(block) > remaining:
            # Keep a small final snippet rather than dropping a relevant source entirely.
            if remaining < 600:
                skipped_budget += 1
                continue
            block = block[:remaining].rstrip() + "\n\n[... контекст обрезан по лимиту RAG_CONTEXT_MAX_CHARS ...]"
            source_truncated = True
            truncated_sources += 1

        blocks.append(block)
        total_used_chars += len(block) + sep_len
        kept = dict(h)
        kept["text_chars"] = len(str(h.get("text") or ""))
        kept["context_chars"] = len(block)
        kept["context_truncated"] = source_truncated
        used_hits.append(kept)

    stats = {
        "requested_hits": len(hits),
        "used_hits": len(used_hits),
        "max_context_chars": max_context_chars,
        "max_source_chars": max_source_chars,
        "context_chars": total_used_chars,
        "original_hit_text_chars": total_original_chars,
        "truncated_sources": truncated_sources,
        "skipped_duplicates": skipped_duplicates,
        "skipped_by_budget": skipped_budget,
    }
    return "\n\n".join(blocks), used_hits, stats


async def _openai_style_chat_completion(
    *,
    base_url: str | None,
    api_key: str | None,
    model: str,
    user_msg: str,
    system: str,
    timeout: float = 120.0,
) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    url = root + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if (api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key}"
    req_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=req_body, headers=headers)
    except httpx.TimeoutException as e:
        raise HTTPException(
            504,
            f"Таймаут LLM ({url}, {timeout} s): {e}",
        ) from e
    except httpx.RequestError as e:
        raise HTTPException(
            502,
            f"Нет связи с LLM ({url}): {type(e).__name__}: {e}. "
            "Для LM Studio: включите Local Server и проверьте RAG_LM_STUDIO_BASE_URL.",
        ) from e

    if r.status_code >= 400:
        raise HTTPException(
            502,
            f"LLM вернул HTTP {r.status_code}: {r.text[:4000]}",
        )

    try:
        data = r.json()
    except json.JSONDecodeError:
        raise HTTPException(
            502,
            f"LLM вернул не-JSON (HTTP {r.status_code}): {r.text[:1200]}",
        ) from None

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return str(data)


def _anthropic_blocks_to_text(message: Any) -> str:
    parts: list[str] = []
    for b in getattr(message, "content", None) or []:
        t = getattr(b, "type", None)
        if t == "text":
            parts.append(str(getattr(b, "text", "") or ""))
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text") or ""))
    return "".join(parts).strip() or str(message)


async def _anthropic_rag_completion(
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    user_msg: str,
    system: str,
) -> str:
    import anthropic

    def _sync() -> Any:
        client = anthropic.Anthropic(api_key=api_key)
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )

    message = await asyncio.to_thread(_sync)
    return _anthropic_blocks_to_text(message)


def _rag_sources_summary(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": h.get("source_path"),
            "heading": h.get("heading"),
            "score": h.get("score"),
            "text_chars": h.get("text_chars"),
            "context_chars": h.get("context_chars"),
            "context_truncated": h.get("context_truncated", False),
        }
        for h in hits
    ]


@app.post("/v1/rag")
async def rag(body: RagRequest) -> dict[str, Any]:
    if state.encoder is None or state.qdrant is None:
        raise HTTPException(503, state.startup_error or "Сервис не готов")

    s = state.settings

    def _do_search() -> dict[str, Any]:
        return run_search(
            SearchRequest(query=body.query, limit=body.limit, score_threshold=None),
        )

    search_payload = await asyncio.to_thread(_do_search)
    hits = search_payload.get("hits") or []
    if not hits:
        return {
            "answer": "В индексе нет релевантных фрагментов по запросу. Проиндексируйте каталог через POST /v1/index.",
            "sources": [],
        }

    try:
        provider = _normalize_rag_provider(body.llm_provider, s)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    max_context_chars = body.max_context_chars or s.rag_context_max_chars
    max_source_chars = body.max_source_chars or s.rag_source_max_chars
    context, context_hits, context_stats = _rag_context_from_hits(
        hits,
        max_context_chars=max_context_chars,
        max_source_chars=max_source_chars,
        dedupe_sources=s.rag_dedupe_sources,
    )
    if not context_hits:
        return {
            "answer": "Релевантные фрагменты найдены, но после применения бюджета контекста не осталось текста для ответа. Увеличьте max_context_chars или проверьте чанки в базе.",
            "sources": [],
            "context_stats": context_stats,
        }
    topic = (body.topic or body.query).strip()
    user_msg = (
        f"Тема ответа: {topic}\n\n"
        f"Используй только следующие источники:\n\n{context}\n\n"
        f"Сформулируй связный ответ с явными ссылками на источники [1], [2], … по ходу текста."
    )

    try:
        return await _rag_answer_with_provider(provider, body, s, context_hits, user_msg, context_stats)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /v1/rag")
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


async def _rag_answer_with_provider(
    provider: str,
    body: RagRequest,
    s: Settings,
    hits: list[dict[str, Any]],
    user_msg: str,
    context_stats: dict[str, Any],
) -> dict[str, Any]:
    if provider == "lm_studio":
        lm_model = _lm_studio_model_for_rag(body, s)
        if not lm_model:
            raise HTTPException(
                501,
                "LM Studio: укажите model в теле запроса или задайте RAG_LM_STUDIO_RAG_MODEL / RAG_SEMANTIC_CHUNK_MODEL.",
            )
        if lm_model in _STOP_LM_MODELS:
            raise HTTPException(
                400,
                f"model={lm_model} — это модель эмбеддингов, а не LLM. "
                f"Укажите имя LLM-модели (например, google/gemma-3-1b)."
            )
        base = (s.lm_studio_base_url or "").strip() or "http://127.0.0.1:1234/v1"
        answer = await _openai_style_chat_completion(
            base_url=base,
            api_key=(s.lm_studio_api_key or "").strip() or None,
            model=lm_model,
            user_msg=user_msg,
            system=SYSTEM_RU,
        )
        note_rag()
        return {
            "answer": answer,
            "sources": _rag_sources_summary(hits),
            "context_stats": context_stats,
            "model": lm_model,
            "llm_provider": "lm_studio",
        }

    if provider == "openai":
        base_url = (s.openai_base_url or "").strip()
        api_key = (s.openai_api_key or "").strip()
        if not base_url and not api_key:
            raise HTTPException(
                501,
                "Провайдер openai: задайте RAG_OPENAI_API_KEY и/или RAG_OPENAI_BASE_URL.",
            )
        model = (body.model or "").strip() or s.default_llm_model
        answer = await _openai_style_chat_completion(
            base_url=base_url or None,
            api_key=api_key or None,
            model=model,
            user_msg=user_msg,
            system=SYSTEM_RU,
        )
        note_rag()
        return {
            "answer": answer,
            "sources": _rag_sources_summary(hits),
            "context_stats": context_stats,
            "model": model,
            "llm_provider": "openai",
        }

    if provider == "anthropic":
        key = (s.anthropic_api_key or "").strip()
        if not key:
            raise HTTPException(501, "Провайдер anthropic: задайте RAG_ANTHROPIC_API_KEY.")
        model = (body.model or "").strip() or s.anthropic_model
        answer = await _anthropic_rag_completion(
            api_key=key,
            model=model,
            max_tokens=s.anthropic_max_tokens,
            user_msg=user_msg,
            system=SYSTEM_RU,
        )
        note_rag()
        return {
            "answer": answer,
            "sources": _rag_sources_summary(hits),
            "context_stats": context_stats,
            "model": model,
            "llm_provider": "anthropic",
        }

    raise HTTPException(500, f"Неизвестный провайдер после нормализации: {provider}")


@app.get("/v1/point-ids")
def point_ids_preview(source_path: str, max_chunk: int = 32) -> dict[str, Any]:
    """Отладка: детерминированные id точек Qdrant для пары (путь, индекс чанка)."""
    out = [stable_point_id(source_path, i) for i in range(max_chunk)]
    return {"source_path": source_path, "ids": out}


def _collection_info_dict(raw: Any) -> dict[str, Any]:
    """Сериализация ответа Qdrant get_collection для JSON."""
    if raw is None:
        return {}
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    md = getattr(raw, "dict", None)
    if callable(md):
        try:
            return md()
        except Exception:
            pass
    return {"detail": str(raw)}


def _truncate_payload_preview(pl: dict[str, Any], text_max: int) -> dict[str, Any]:
    out = dict(pl)
    t = out.get("text")
    if isinstance(t, str) and len(t) > text_max:
        out["text"] = t[:text_max] + "…"
        out["text_truncated"] = True
    return out


def _qdrant_collection_names() -> list[str]:
    if state.qdrant is None:
        return []
    raw = state.qdrant.get_collections()
    collections = getattr(raw, "collections", None) or []
    names: list[str] = []
    for c in collections:
        name = getattr(c, "name", None)
        if name:
            names.append(str(name))
    return sorted(names)


def _qdrant_count(collection_name: str) -> int | None:
    if state.qdrant is None:
        return None
    try:
        c = state.qdrant.count(collection_name=collection_name, exact=False)
        return int(getattr(c, "count", c) or 0)
    except Exception:
        return None


def _collection_overview(name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "collection_name": name,
        "active": name == state.settings.collection_name,
        "points_count": _qdrant_count(name),
        "status": None,
        "vectors_count": None,
        "error": None,
    }
    if state.qdrant is None:
        info["error"] = "Qdrant не подключён"
        return info
    try:
        raw = state.qdrant.get_collection(name)
        dumped = _collection_info_dict(raw)
        info["status"] = dumped.get("status")
        info["vectors_count"] = dumped.get("vectors_count")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _sources_summary_from_points(points: list[Any]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for p in points:
        pl = getattr(p, "payload", None) or {}
        if not isinstance(pl, dict):
            continue
        source = str(pl.get("source_path") or pl.get("file_name") or "unknown")
        item = by_source.setdefault(
            source,
            {
                "source_path": source,
                "file_name": pl.get("file_name"),
                "parent_dir": pl.get("parent_dir"),
                "corpus_root": pl.get("corpus_root"),
                "chunks": 0,
                "text_chars": 0,
                "indexed_at_unix": pl.get("indexed_at_unix"),
                "headings_sample": [],
            },
        )
        item["chunks"] += 1
        text_chars = pl.get("chunk_text_chars")
        if not isinstance(text_chars, int):
            text_chars = len(str(pl.get("text") or ""))
        item["text_chars"] += text_chars
        indexed_at = pl.get("indexed_at_unix")
        if isinstance(indexed_at, (int, float)):
            current = item.get("indexed_at_unix")
            item["indexed_at_unix"] = max(float(current or 0), float(indexed_at))
        heading = str(pl.get("heading") or "").strip()
        sample = item["headings_sample"]
        if heading and heading not in sample and len(sample) < 5:
            sample.append(heading)
    return sorted(by_source.values(), key=lambda x: (-int(x["chunks"]), str(x["source_path"])))


@app.get("/v1/qdrant/collections")
def qdrant_collections_overview() -> dict[str, Any]:
    """Список коллекций Qdrant, чтобы видеть отдельные базы знаний и их размер."""
    if state.qdrant is None:
        raise HTTPException(503, state.startup_error or "Qdrant не подключён")
    try:
        names = _qdrant_collection_names()
    except Exception as e:
        raise HTTPException(502, f"get_collections: {e}") from e
    return {
        "active_collection": state.settings.collection_name,
        "collections_count": len(names),
        "collections": [_collection_overview(name) for name in names],
    }


@app.get("/v1/qdrant/sources")
def qdrant_sources_overview(
    collection_name: str | None = Query(default=None, description="По умолчанию активная коллекция"),
    scan_limit: int = Query(5000, ge=1, le=50000),
) -> dict[str, Any]:
    """Агрегация активной базы по исходным файлам: чанки, символы, заголовки."""
    if state.qdrant is None:
        raise HTTPException(503, state.startup_error or "Qdrant не подключён")
    name = (collection_name or state.settings.collection_name).strip()
    points: list[Any] = []
    offset: Any = None
    try:
        while len(points) < scan_limit:
            batch, offset = state.qdrant.scroll(
                collection_name=name,
                limit=min(256, scan_limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None or not batch:
                break
    except Exception as e:
        raise HTTPException(502, f"scroll: {e}") from e

    sources = _sources_summary_from_points(points)
    return {
        "collection_name": name,
        "points_scanned": len(points),
        "scan_limit": scan_limit,
        "truncated": offset is not None,
        "sources_count": len(sources),
        "sources": sources,
        "note": "Если truncated=true, увеличьте scan_limit или используйте пагинацию /v1/qdrant/points для полного обхода.",
    }


@app.get("/v1/qdrant/collection")
def qdrant_collection_meta() -> dict[str, Any]:
    """Сводка по коллекции Qdrant: схема векторов, подсчёт точек, статус."""
    if state.qdrant is None:
        raise HTTPException(503, state.startup_error or "Qdrant не подключён")
    s = state.settings
    name = s.collection_name
    try:
        info = state.qdrant.get_collection(name)
    except Exception as e:
        raise HTTPException(502, f"get_collection: {e}") from e
    cnt = None
    try:
        c = state.qdrant.count(collection_name=name, exact=False)
        cnt = int(getattr(c, "count", c) or 0)
    except Exception:
        pass
    return {
        "collection_name": name,
        "points_count": cnt,
        "info": _collection_info_dict(info),
    }


@app.get("/v1/qdrant/points")
def qdrant_points_browse(
    limit: int = Query(25, ge=1, le=200),
    text_preview_chars: int = Query(400, ge=80, le=4000),
    offset: str | None = Query(default=None, description="next_page_offset из предыдущего ответа"),
) -> dict[str, Any]:
    """
    Срез точек из Qdrant (scroll): наглядный просмотр индекса без векторов.
    Для следующей страницы передайте next_page_offset обратно в offset.
    """
    if state.qdrant is None:
        raise HTTPException(503, state.startup_error or "Qdrant не подключён")
    s = state.settings
    name = s.collection_name
    try:
        points, next_offset = state.qdrant.scroll(
            collection_name=name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(502, f"scroll: {e}") from e

    rows: list[dict[str, Any]] = []
    for p in points:
        pl = getattr(p, "payload", None) or {}
        if not isinstance(pl, dict):
            pl = {}
        pid = getattr(p, "id", None)
        preview = _truncate_payload_preview(pl, text_preview_chars)
        rows.append(
            {
                "id": str(pid) if pid is not None else None,
                "payload": preview,
            }
        )

    has_more = next_offset is not None
    safe_next: str | int | None
    if next_offset is None:
        safe_next = None
    elif isinstance(next_offset, (str, int)):
        safe_next = next_offset
    else:
        safe_next = str(next_offset)
    return {
        "collection_name": name,
        "limit": limit,
        "offset": offset,
        "returned": len(rows),
        "has_more": has_more,
        "next_page_offset": safe_next,
        "note": "Для следующей страницы передайте next_page_offset в query-параметр offset.",
        "points": rows,
    }
