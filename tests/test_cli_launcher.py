from __future__ import annotations

import sys
from pathlib import Path

import llmmd
from llmmd_core.cli import build_internal_parser, build_lite_parser


def test_build_mcp_config_uses_llmmd_rag_base_url():
    root = Path("C:/work/llmmd")
    cfg = llmmd.build_mcp_config(
        python_exe="C:/Python/python.exe",
        rag_url="http://127.0.0.1:8765/",
        root=root,
    )

    server = cfg["mcpServers"]["llmmd-rag"]
    assert server["command"] == "C:/Python/python.exe"
    assert server["args"] == [str(root / "claude_mcp" / "llmmd_rag_server.py")]
    assert server["env"]["LLMMD_RAG_BASE_URL"] == "http://127.0.0.1:8765"
    assert server["env"]["PYTHONPATH"] == str(root)


def test_lite_parser_defaults_to_run():
    parser = build_lite_parser()
    args = parser.parse_args([])
    assert args.command == "run"
    assert args.skip_qdrant is False


def test_lite_parser_has_setup_and_run_only():
    parser = build_lite_parser()
    args = parser.parse_args(["setup", "--skip-qdrant"])
    assert args.command == "setup"
    assert args.skip_qdrant is True


def test_internal_parser_has_rag():
    parser = build_internal_parser()
    args = parser.parse_args(["rag", "--port", "8766"])
    assert args.command == "rag"
    assert args.port == 8766


def test_internal_parser_mcp_config():
    parser = build_internal_parser()
    args = parser.parse_args(["--no-install", "mcp-config", "--python", sys.executable])
    assert args.no_install is True
    assert args.command == "mcp-config"
    assert args.python == sys.executable


def test_missing_imports_returns_list_for_known_extra():
    missing = llmmd._missing_imports("mcp")
    assert isinstance(missing, list)
