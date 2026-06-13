from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_VERSION = "0.5.0"

EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "gui": ("markitdown", "openai", "fitz"),
    "rag": ("fastapi", "uvicorn", "qdrant_client", "sentence_transformers", "pydantic_settings", "numpy"),
    "lmstudio": ("fastapi", "uvicorn", "httpx", "pydantic"),
    "web": ("streamlit", "requests", "psutil", "GPUtil"),
    "mcp": ("mcp", "httpx"),
}

VALID_EXTRAS = frozenset({"gui", "rag", "lmstudio", "web", "mcp", "dev", "all"})
DEFAULT_CONFIG_REL = Path("config") / "llmmd.yaml"
DEFAULT_DOCTOR_ENV_KEYS = ("RAG_QDRANT_URL", "RAG_CORPUS_ROOT", "LLMMD_RAG_BASE_URL", "LLMMD_CONFIG")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_file: Path

    @property
    def gui_webui_script(self) -> Path:
        return self.root / "gui" / "webui.py"

    @property
    def mcp_server_script(self) -> Path:
        return self.root / "claude_mcp" / "llmmd_rag_server.py"


@dataclass(frozen=True)
class ServiceEndpoint:
    host: str | None = None
    port: int | None = None
    reload: bool = False


@dataclass(frozen=True)
class QdrantDockerConfig:
    compose_file: Path
    url: str = "http://localhost:6333"
    health_paths: tuple[str, ...] = ("/healthz", "/readyz", "/collections")
    wait_timeout_s: float = 60.0
    wait_interval_s: float = 1.0
    logs_tail: int = 120


@dataclass(frozen=True)
class McpConfig:
    rag_url: str = "http://127.0.0.1:8765"


@dataclass(frozen=True)
class LauncherConfig:
    paths: ProjectPaths
    web: ServiceEndpoint = field(default_factory=lambda: ServiceEndpoint(host="127.0.0.1", port=8501))
    rag: ServiceEndpoint = field(default_factory=ServiceEndpoint)
    lmstudio: ServiceEndpoint = field(default_factory=lambda: ServiceEndpoint(host="127.0.0.1", port=8790))
    qdrant: QdrantDockerConfig | None = None
    mcp: McpConfig = field(default_factory=McpConfig)
    doctor_env_keys: tuple[str, ...] = DEFAULT_DOCTOR_ENV_KEYS


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_config_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / DEFAULT_CONFIG_REL


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        if path == default_config_path().resolve():
            return {}
        raise RuntimeError(f"PyYAML is required to read launcher config: {path}") from None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Launcher config must be a mapping: {path}")
    return raw


def _section(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _endpoint(data: dict[str, Any], *, default_host: str | None, default_port: int | None) -> ServiceEndpoint:
    return ServiceEndpoint(
        host=data.get("host", default_host),
        port=_int_or_none(data.get("port", default_port)),
        reload=bool(data.get("reload", False)),
    )


def _resolve_project_path(root: Path, value: Any, default: Path) -> Path:
    raw = Path(str(value)) if value not in (None, "") else default
    if raw.is_absolute():
        return raw
    return root / raw


def _health_paths(data: dict[str, Any], default: list[str]) -> tuple[str, ...]:
    health_paths_raw = data.get("health_paths") or default
    return tuple(str(p if str(p).startswith("/") else f"/{p}") for p in health_paths_raw)


def _qdrant_config(root: Path, data: dict[str, Any]) -> QdrantDockerConfig:
    return QdrantDockerConfig(
        compose_file=_resolve_project_path(root, data.get("compose_file"), Path("rag_service/docker-compose.qdrant.yml")),
        url=str(data.get("url") or "http://localhost:6333").rstrip("/"),
        health_paths=_health_paths(data, ["/healthz", "/readyz", "/collections"]),
        wait_timeout_s=_float(data.get("wait_timeout_s"), 60.0),
        wait_interval_s=_float(data.get("wait_interval_s"), 1.0),
        logs_tail=int(data.get("logs_tail") or 120),
    )


def load_launcher_config(path: str | Path | None = None) -> LauncherConfig:
    root = repo_root()
    selected = Path(path or os.environ.get("LLMMD_CONFIG") or default_config_path(root)).expanduser()
    if not selected.is_absolute():
        selected = root / selected
    selected = selected.resolve()
    data = _read_yaml(selected)
    services = _section(data, "services")
    doctor = _section(data, "doctor")
    paths = ProjectPaths(root=root, config_file=selected)

    qdrant = _qdrant_config(root, _section(services, "qdrant"))
    return LauncherConfig(
        paths=paths,
        web=_endpoint(_section(services, "web"), default_host="127.0.0.1", default_port=8501),
        rag=_endpoint(_section(services, "rag"), default_host=None, default_port=None),
        lmstudio=_endpoint(_section(services, "lmstudio"), default_host="127.0.0.1", default_port=8790),
        qdrant=qdrant,
        mcp=McpConfig(rag_url=str(_section(services, "mcp").get("rag_url") or "http://127.0.0.1:8765").rstrip("/")),
        doctor_env_keys=tuple(str(v) for v in (doctor.get("env_keys") or DEFAULT_DOCTOR_ENV_KEYS)),
    )
