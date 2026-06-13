from __future__ import annotations

import json
from pathlib import Path


def build_mcp_config(*, python_exe: str, rag_url: str, root: Path) -> dict[str, object]:
    script = root / "claude_mcp" / "llmmd_rag_server.py"
    return {
        "mcpServers": {
            "llmmd-rag": {
                "command": python_exe,
                "args": [str(script)],
                "env": {
                    "LLMMD_RAG_BASE_URL": rag_url.rstrip("/"),
                    "PYTHONPATH": str(root),
                },
            }
        }
    }


def mcp_config_json(*, python_exe: str, rag_url: str, root: Path) -> str:
    return json.dumps(
        build_mcp_config(python_exe=python_exe, rag_url=rag_url, root=root),
        indent=2,
        ensure_ascii=False,
    )


def write_mcp_config(path: Path, *, python_exe: str, rag_url: str, root: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mcp_config_json(python_exe=python_exe, rag_url=rag_url, root=root) + "\n", encoding="utf-8")
    return path
