from __future__ import annotations

from pathlib import Path

from llmmd_core.config import load_launcher_config


def test_launcher_config_loads_yaml_overrides(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "llmmd.yaml"
    cfg_path.write_text(
        """
services:
  web:
    host: 0.0.0.0
    port: 8510
  qdrant:
    compose_file: rag_service/docker-compose.qdrant.yml
    url: http://localhost:6333
    health_paths: [/readyz]
    wait_timeout_s: 3
doctor:
  env_keys: [RAG_QDRANT_URL]
""",
        encoding="utf-8",
    )

    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    cfg = load_launcher_config(cfg_path)

    assert cfg.paths.config_file == cfg_path.resolve()
    assert cfg.web.host == "0.0.0.0"
    assert cfg.web.port == 8510
    assert cfg.qdrant is not None
    assert cfg.qdrant.health_paths == ("/readyz",)
    assert cfg.qdrant.wait_timeout_s == 3
    assert cfg.doctor_env_keys == ("RAG_QDRANT_URL",)


def test_launcher_config_has_qdrant_section(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    cfg = load_launcher_config()

    assert cfg.qdrant is not None
    assert cfg.qdrant.url == "http://localhost:6333"
    assert "qdrant" in str(cfg.qdrant.compose_file)
