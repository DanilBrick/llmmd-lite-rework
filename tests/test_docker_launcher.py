from __future__ import annotations

from pathlib import Path

from llmmd_core.config import QdrantDockerConfig
from llmmd_core.docker import compose_command, qdrant_probe_urls


def test_compose_command_uses_configured_compose_file():
    cfg = QdrantDockerConfig(compose_file=Path("rag_service/docker-compose.qdrant.yml"))

    cmd = compose_command(cfg, "up", "-d")

    assert cmd[:3] == ["docker", "compose", "-f"]
    assert cmd[3] == str(cfg.compose_file)
    assert cmd[4:] == ["up", "-d"]


def test_qdrant_probe_urls_are_derived_from_config():
    cfg = QdrantDockerConfig(
        compose_file=Path("compose.yml"),
        url="http://localhost:6333/",
        health_paths=("/healthz", "/collections"),
    )

    assert qdrant_probe_urls(cfg) == ["http://localhost:6333/healthz", "http://localhost:6333/collections"]
