from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .commands import (
    cmd_doctor,
    cmd_info,
    cmd_lmstudio,
    cmd_mcp,
    cmd_mcp_config,
    cmd_qdrant,
    cmd_rag,
    cmd_run,
    cmd_runall,
    cmd_setup_lite,
    cmd_web,
)
from .config import PROJECT_VERSION, LauncherConfig, load_launcher_config

INTERNAL_COMMANDS = frozenset(
    {
        "rag",
        "web",
        "lmstudio",
        "runall",
        "qdrant",
        "mcp",
        "mcp-config",
        "doctor",
        "info",
    }
)


def _config_path_from_argv(argv: list[str]) -> str | None:
    for i, item in enumerate(argv):
        if item == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith("--config="):
            return item.split("=", 1)[1]
    return None


def _command_token(argv: list[str]) -> str | None:
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--config":
            i += 2
            continue
        if item.startswith("--config="):
            i += 1
            continue
        if item in ("--no-install", "--version", "-h", "--help"):
            i += 1
            continue
        if item.startswith("-"):
            i += 1
            continue
        return item
    return None


def _is_internal_invocation(argv: list[str]) -> bool:
    return _command_token(argv) in INTERNAL_COMMANDS


def _lite_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["run"]
    first = _command_token(argv)
    if first in (None, "run", "setup"):
        return argv
    if first.startswith("-"):
        return ["run", *argv]
    return argv


def build_lite_parser(config: LauncherConfig | None = None) -> argparse.ArgumentParser:
    config = config or load_launcher_config()
    parser = argparse.ArgumentParser(
        prog="llmmd.py",
        description="llmmd lite — подготовка и запуск на ноутбуке.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python llmmd.py setup\n"
            "  python llmmd.py\n"
            "  python llmmd.py run\n"
        ),
    )
    parser.add_argument("--config", default=str(config.paths.config_file), help="Путь к config/llmmd.yaml")
    parser.add_argument("--no-install", action="store_true", help="Не ставить зависимости автоматически.")
    parser.add_argument("--version", action="version", version=f"llmmd {PROJECT_VERSION}")
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="Установка, проверка, Qdrant Docker")
    p_setup.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Не поднимать Qdrant (удалённая БД или Docker недоступен).",
    )
    p_setup.set_defaults(func=cmd_setup_lite)

    p_run = sub.add_parser("run", help="Qdrant + RAG + панель с чатом")
    p_run.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Не запускать Qdrant Docker (уже запущен или удалённый).",
    )
    p_run.set_defaults(func=cmd_run)

    parser.set_defaults(func=cmd_run, command="run", skip_qdrant=False)
    return parser


def build_internal_parser(config: LauncherConfig | None = None) -> argparse.ArgumentParser:
    config = config or load_launcher_config()
    parser = argparse.ArgumentParser(prog="llmmd.py", description="llmmd internal service launcher.")
    parser.add_argument("--config", default=str(config.paths.config_file))
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--version", action="version", version=f"llmmd {PROJECT_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_web = sub.add_parser("web")
    p_web.add_argument("--host", default=config.web.host or "127.0.0.1")
    p_web.add_argument("--port", type=int, default=config.web.port or 8501)
    p_web.set_defaults(func=cmd_web)

    p_rag = sub.add_parser("rag")
    p_rag.add_argument("--host", default=config.rag.host)
    p_rag.add_argument("--port", type=int, default=config.rag.port)
    p_rag.add_argument("--reload", action="store_true", default=config.rag.reload)
    p_rag.set_defaults(func=cmd_rag)

    p_lm = sub.add_parser("lmstudio")
    p_lm.add_argument("--host", default=config.lmstudio.host)
    p_lm.add_argument("--port", type=int, default=config.lmstudio.port)
    p_lm.set_defaults(func=cmd_lmstudio)

    p_all = sub.add_parser("runall")
    p_all.add_argument("--skip-qdrant", action="store_true")
    p_all.add_argument("--qdrant-no-wait", action="store_true")
    p_all.add_argument("--qdrant-pull", action="store_true")
    p_all.add_argument("--rag-host", default=None)
    p_all.add_argument("--rag-port", type=int, default=None)
    p_all.add_argument("--rag-reload", action="store_true")
    p_all.add_argument("--web-host", default=config.web.host or "127.0.0.1")
    wp_def = config.web.port if config.web.port is not None else 8501
    p_all.add_argument("--web-port", type=int, default=wp_def)
    p_all.set_defaults(func=cmd_runall)

    p_qd = sub.add_parser("qdrant")
    p_qd.add_argument("action", choices=("up", "down", "restart", "status", "ps", "logs"), nargs="?", default="up")
    p_qd.add_argument("--no-wait", action="store_true")
    p_qd.add_argument("--pull", action="store_true")
    p_qd.add_argument("--volumes", action="store_true")
    p_qd.add_argument("--follow", "-f", action="store_true")
    default_tail = config.qdrant.logs_tail if config.qdrant is not None else 120
    p_qd.add_argument("--tail", type=int, default=default_tail)
    p_qd.set_defaults(func=cmd_qdrant)

    p_mcp = sub.add_parser("mcp")
    p_mcp.add_argument("--rag-url", default=config.mcp.rag_url)
    p_mcp.set_defaults(func=cmd_mcp)

    p_mcp_cfg = sub.add_parser("mcp-config")
    p_mcp_cfg.add_argument("--rag-url", default=config.mcp.rag_url)
    p_mcp_cfg.add_argument("--python", default=None)
    p_mcp_cfg.add_argument("--write", default=None)
    p_mcp_cfg.set_defaults(func=cmd_mcp_config)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    sub.add_parser("info").set_defaults(func=cmd_info)
    return parser


def build_parser(config: LauncherConfig | None = None) -> argparse.ArgumentParser:
    return build_lite_parser(config)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config = load_launcher_config(_config_path_from_argv(raw_argv))
    internal = _is_internal_invocation(raw_argv)
    parser = build_internal_parser(config) if internal else build_lite_parser(config)
    parse_argv = raw_argv if internal else _lite_argv(raw_argv)
    args = parser.parse_args(parse_argv)
    if args.config != str(config.paths.config_file):
        config = load_launcher_config(Path(args.config))
    args.launcher_config = config
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"[llmmd] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
