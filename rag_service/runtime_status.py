"""
Потокобезопасное состояние для мониторинга: сервис, последняя/текущая индексация, вспомогательные метрики.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ServiceRuntimeSnapshot:
    """Состояние ядра после старта (обновляется из lifespan)."""

    phase: str = "starting"  # starting | ready | degraded
    startup_error: Optional[str] = None
    started_at_unix: float = 0.0
    ready_at_unix: float = 0.0
    embedding_model: str = ""
    hybrid_enabled: bool = False
    dense_dim: int = 0
    qdrant_url: str = ""
    collection_name: str = ""


@dataclass
class IndexJobSnapshot:
    job_id: str = ""
    phase: str = "idle"  # idle | running | completed | failed
    wait_mode: str = ""  # sync | async
    started_at_unix: float = 0.0
    finished_at_unix: float = 0.0
    corpus_root: Optional[str] = None
    glob_pattern: str = ""
    chunking_mode: str = ""
    recreate_collection: bool = False
    files_total: int = 0
    files_done: int = 0
    files_skipped_empty: int = 0
    chunks_upserted: int = 0
    current_file: Optional[str] = None
    current_stage: str = ""  # scan | index_file | done
    seconds_elapsed: float = 0.0
    last_error: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    result_summary: Optional[dict[str, Any]] = None


@dataclass
class ActivitySnapshot:
    last_search_at_unix: float = 0.0
    last_search_query_len: int = 0
    last_rag_at_unix: float = 0.0
    searches_total: int = 0
    rag_total: int = 0


_lock = threading.Lock()
_service = ServiceRuntimeSnapshot(started_at_unix=time.time())
_index_job = IndexJobSnapshot()
_activity = ActivitySnapshot()


def service_snapshot() -> ServiceRuntimeSnapshot:
    with _lock:
        return ServiceRuntimeSnapshot(**asdict(_service))


def set_service_ready(
    *,
    startup_error: Optional[str],
    embedding_model: str,
    hybrid: bool,
    dense_dim: int,
    qdrant_url: str,
    collection_name: str,
) -> None:
    with _lock:
        global _service
        now = time.time()
        _service.startup_error = startup_error
        _service.embedding_model = embedding_model
        _service.hybrid_enabled = hybrid
        _service.dense_dim = dense_dim
        _service.qdrant_url = qdrant_url
        _service.collection_name = collection_name
        _service.ready_at_unix = now
        if startup_error:
            _service.phase = "degraded"
        else:
            _service.phase = "ready"


def set_service_starting() -> None:
    with _lock:
        global _service
        _service.phase = "starting"
        _service.started_at_unix = time.time()
        _service.startup_error = None
        _service.ready_at_unix = 0.0


def index_job_snapshot() -> IndexJobSnapshot:
    with _lock:
        snap = IndexJobSnapshot(**asdict(_index_job))
        if snap.phase == "running" and snap.started_at_unix:
            snap.seconds_elapsed = round(time.time() - snap.started_at_unix, 2)
        elif snap.finished_at_unix and snap.started_at_unix:
            snap.seconds_elapsed = round(snap.finished_at_unix - snap.started_at_unix, 3)
        return snap


def _replace_index_job(s: IndexJobSnapshot) -> None:
    global _index_job  # noqa: PLW0603
    _index_job = s


def index_job_begin(
    job_id: str,
    *,
    wait_mode: str,
    corpus_root: str,
    glob_pattern: str,
    chunking_mode: str,
    recreate_collection: bool,
    files_total: int,
) -> None:
    with _lock:
        _replace_index_job(
            IndexJobSnapshot(
                job_id=job_id,
                phase="running",
                wait_mode=wait_mode,
                started_at_unix=time.time(),
                corpus_root=corpus_root,
                glob_pattern=glob_pattern,
                chunking_mode=chunking_mode,
                recreate_collection=recreate_collection,
                files_total=files_total,
                files_done=0,
                files_skipped_empty=0,
                chunks_upserted=0,
                current_file=None,
                current_stage="scan",
                last_error=None,
                errors=[],
                result_summary=None,
            )
        )


def index_job_progress(
    *,
    files_done: int,
    chunks_upserted: int,
    current_file: Optional[str],
    current_stage: str,
    files_skipped_empty: int = 0,
) -> None:
    with _lock:
        global _index_job
        _index_job.files_done = files_done
        _index_job.chunks_upserted = chunks_upserted
        _index_job.current_file = current_file
        _index_job.current_stage = current_stage
        _index_job.files_skipped_empty = files_skipped_empty


def index_job_append_error(msg: str) -> None:
    with _lock:
        global _index_job
        _index_job.errors.append(msg)
        if len(_index_job.errors) > 200:
            _index_job.errors = _index_job.errors[-200:]
        _index_job.last_error = msg


def index_job_finish_ok(summary: dict[str, Any]) -> None:
    with _lock:
        global _index_job
        _index_job.phase = "completed"
        _index_job.finished_at_unix = time.time()
        _index_job.current_file = None
        _index_job.current_stage = "done"
        _index_job.result_summary = summary


def index_job_finish_fail(message: str) -> None:
    with _lock:
        global _index_job
        _index_job.phase = "failed"
        _index_job.finished_at_unix = time.time()
        _index_job.current_file = None
        _index_job.current_stage = "done"
        _index_job.last_error = message


def index_job_reset_idle() -> None:
    with _lock:
        _replace_index_job(IndexJobSnapshot())


def is_index_job_running() -> bool:
    with _lock:
        return _index_job.phase == "running"


def note_search(query: str) -> None:
    with _lock:
        global _activity
        _activity.last_search_at_unix = time.time()
        _activity.last_search_query_len = len(query or "")
        _activity.searches_total += 1


def note_rag() -> None:
    with _lock:
        global _activity
        _activity.last_rag_at_unix = time.time()
        _activity.rag_total += 1


def activity_snapshot() -> ActivitySnapshot:
    with _lock:
        return ActivitySnapshot(**asdict(_activity))


def new_job_id() -> str:
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
