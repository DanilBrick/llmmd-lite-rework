#!/usr/bin/env python3
"""Compatibility wrapper for the decomposed llmmd launcher."""

from __future__ import annotations

from pathlib import Path

from llmmd_core.cli import build_parser, main
from llmmd_core.config import PROJECT_VERSION, load_launcher_config, repo_root
from llmmd_core.dependencies import ensure_runtime as _ensure_runtime
from llmmd_core.dependencies import missing_imports as _missing_imports
from llmmd_core.mcp import build_mcp_config as _build_mcp_config


def ensure_runtime(extra, *, no_install: bool = False) -> None:
    _ensure_runtime(extra, root=repo_root(), no_install=no_install)


def build_mcp_config(*, python_exe: str, rag_url: str, root: Path | None = None) -> dict[str, object]:
    return _build_mcp_config(python_exe=python_exe, rag_url=rag_url, root=root or repo_root())


__all__ = [
    "PROJECT_VERSION",
    "_missing_imports",
    "build_mcp_config",
    "build_parser",
    "ensure_runtime",
    "load_launcher_config",
    "main",
    "repo_root",
]


if __name__ == "__main__":
    raise SystemExit(main())
