from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import PROJECT_VERSION, LauncherConfig
from .dependencies import ensure_runtime, install_extras, missing_imports
from .mcp import mcp_config_json, write_mcp_config
from .docker import (
    docker_status,
    qdrant_down,
    qdrant_logs,
    qdrant_ps,
    qdrant_status,
    qdrant_up,
)
def _stop_runall_children(processes: list[subprocess.Popen], *, grace_s: float = 8.0, per_proc_grace_s: float = 3.0) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + grace_s
    for proc in processes:
        if proc.poll() is not None:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            proc.wait(timeout=min(per_proc_grace_s, remaining))
        except subprocess.TimeoutExpired:
            pass
    for proc in processes:
        if proc.poll() is None:
            proc.kill()


def cmd_runall(args: argparse.Namespace) -> int:
    """Bring up Qdrant (unless skipped), then RAG and Streamlit web as sibling processes."""
    cfg = _cfg(args)
    ensure_runtime(("gui", "rag", "web"), root=cfg.paths.root, no_install=args.no_install)

    if not args.skip_qdrant:
        if cfg.qdrant is None:
            print("[llmmd] runall: qdrant is not configured; use --skip-qdrant when using remote Qdrant.")
            return 2
        code = qdrant_up(cfg.qdrant, root=cfg.paths.root, wait=not args.qdrant_no_wait, pull=args.qdrant_pull)
        if code:
            return code

    launcher = cfg.paths.root / "llmmd.py"
    base = [sys.executable, str(launcher), "--config", args.config]
    if args.no_install:
        base.append("--no-install")

    from rag_service.config import Settings

    s = Settings()
    rag_host_val = args.rag_host.strip() if (args.rag_host and args.rag_host.strip()) else s.api_host
    rag_port_val = args.rag_port if args.rag_port is not None else s.api_port
    web_host_val = (
        args.web_host.strip()
        if (args.web_host and args.web_host.strip())
        else (cfg.web.host or "127.0.0.1")
    )
    web_port_val = args.web_port if args.web_port is not None else (cfg.web.port or 8501)

    rag_argv = [*base, "rag", "--host", rag_host_val, "--port", str(rag_port_val)]
    if args.rag_reload:
        rag_argv.append("--reload")
    web_argv = [*base, "web", "--host", web_host_val, "--port", str(web_port_val)]

    rag_p = subprocess.Popen(rag_argv, cwd=cfg.paths.root)
    web_p = subprocess.Popen(web_argv, cwd=cfg.paths.root)
    children = [rag_p, web_p]
    labels = ("rag", "web")

    print("[llmmd] runall: RAG and web are starting.")
    print(f"[llmmd]   - RAG API: http://{rag_host_val}:{rag_port_val}/")
    print(f"[llmmd]   - Панель (чат в UI): http://{web_host_val}:{web_port_val}/")
    print("[llmmd] Press Ctrl+C to stop RAG and web (Docker services stay running).")

    rc = 0
    try:
        while True:
            child_exited = False
            for proc, svc in zip(children, labels, strict=True):
                code = proc.poll()
                if code is not None:
                    print(f"[llmmd] runall: {svc} exited with code {code}.", file=sys.stderr)
                    rc = code if code is not None else 1
                    child_exited = True
                    break
            if child_exited:
                break
            time.sleep(0.35)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        print("[llmmd] runall: stopping RAG and web…", file=sys.stderr)
        rc = 130
    finally:
        _stop_runall_children(children)
    return rc


def _cfg(args: argparse.Namespace) -> LauncherConfig:
    return args.launcher_config


def _add_repo_to_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# Streamlit defaults to 200 MB for uploads and WebSocket payloads.
_STREAMLIT_MAX_MB = "10240"


def _streamlit_web_cmd(cfg: LauncherConfig, *, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(cfg.paths.gui_webui_script),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.maxUploadSize",
        _STREAMLIT_MAX_MB,
        "--server.maxMessageSize",
        _STREAMLIT_MAX_MB,
    ]


def cmd_web(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ensure_runtime(("gui", "web"), root=cfg.paths.root, no_install=args.no_install)
    cmd = _streamlit_web_cmd(cfg, host=args.host, port=args.port)
    return subprocess.call(cmd, cwd=cfg.paths.root)


def cmd_rag(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ensure_runtime("rag", root=cfg.paths.root, no_install=args.no_install)
    import uvicorn

    from rag_service.config import Settings

    s = Settings()
    uvicorn.run(
        "rag_service.main:app",
        host=args.host or s.api_host,
        port=args.port or s.api_port,
        reload=args.reload,
    )
    return 0


def cmd_lmstudio(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ensure_runtime("lmstudio", root=cfg.paths.root, no_install=args.no_install)
    import uvicorn

    host = args.host or os.environ.get("LMSTUDIO_AUTOLOAD_HOST") or cfg.lmstudio.host or "127.0.0.1"
    port = args.port or int(os.environ.get("LMSTUDIO_AUTOLOAD_PORT") or cfg.lmstudio.port or 8790)
    uvicorn.run("lmstudio_autoload.api:app", host=host, port=port)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ensure_runtime("mcp", root=cfg.paths.root, no_install=args.no_install)
    os.environ.setdefault("LLMMD_RAG_BASE_URL", args.rag_url.rstrip("/"))
    from claude_mcp.llmmd_rag_server import mcp

    mcp.run(transport="stdio")
    return 0


def cmd_mcp_config(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    python_exe = args.python or sys.executable
    rag_url = args.rag_url.rstrip("/")
    if args.write:
        target = write_mcp_config(Path(args.write), python_exe=python_exe, rag_url=rag_url, root=cfg.paths.root)
        print(f"Wrote MCP config: {target}")
    else:
        print(mcp_config_json(python_exe=python_exe, rag_url=rag_url, root=cfg.paths.root))
    return 0


def cmd_qdrant(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    qd = cfg.qdrant
    if qd is None:
        raise RuntimeError("Qdrant Docker config is not defined")
    action = args.action
    if action == "up":
        return qdrant_up(qd, root=cfg.paths.root, wait=not args.no_wait, pull=args.pull)
    if action == "down":
        return qdrant_down(qd, root=cfg.paths.root, volumes=args.volumes)
    if action == "restart":
        code = qdrant_down(qd, root=cfg.paths.root, volumes=False)
        if code:
            return code
        return qdrant_up(qd, root=cfg.paths.root, wait=not args.no_wait, pull=args.pull)
    if action == "logs":
        return qdrant_logs(qd, root=cfg.paths.root, follow=args.follow, tail=args.tail)
    if action == "ps":
        return qdrant_ps(qd, root=cfg.paths.root)
    if action == "status":
        return qdrant_status(qd, root=cfg.paths.root)
    raise RuntimeError(f"Unknown qdrant action: {action}")


def cmd_setup(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    install_extras(getattr(args, "extras", None) or ["all"], root=cfg.paths.root)
    return 0


def cmd_setup_lite(args: argparse.Namespace) -> int:
    """Один шаг подготовки: зависимости, doctor, Qdrant Docker."""
    cfg = _cfg(args)
    print("[llmmd] setup: installing dependencies…")
    install_extras(["all"], root=cfg.paths.root)
    print("[llmmd] setup: doctor")
    cmd_doctor(args)
    if not args.skip_qdrant and cfg.qdrant is not None:
        ds = docker_status(root=cfg.paths.root)
        if ds.ok:
            print("[llmmd] setup: starting Qdrant (Docker)…")
            code = qdrant_up(cfg.qdrant, root=cfg.paths.root, wait=True, pull=False)
            if code:
                return code
        else:
            print(f"[llmmd] setup: skip Qdrant — {ds.message}", file=sys.stderr)
    print("[llmmd] setup: готово. Запуск: python llmmd.py")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Запуск полного стека для ноутбука."""
    from rag_service.config import Settings

    cfg = _cfg(args)
    s = Settings()
    runall = argparse.Namespace(
        launcher_config=cfg,
        config=args.config,
        no_install=args.no_install,
        skip_qdrant=args.skip_qdrant,
        qdrant_no_wait=False,
        qdrant_pull=False,
        rag_host=None,
        rag_port=None,
        rag_reload=False,
        web_host=cfg.web.host or "127.0.0.1",
        web_port=cfg.web.port or 8501,
    )
    _ = s  # Settings() ensures defaults are loaded before subprocess RAG starts.
    return cmd_runall(runall)


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ds = docker_status(root=cfg.paths.root)
    rows: list[tuple[str, str]] = [
        ("project", str(cfg.paths.root)),
        ("config", str(cfg.paths.config_file)),
        ("python", sys.executable),
        ("version", sys.version.split()[0]),
        ("docker", ds.message),
    ]
    if cfg.qdrant is not None:
        rows.extend(
            [
                ("qdrant.compose", str(cfg.qdrant.compose_file)),
                ("qdrant.url", cfg.qdrant.url),
            ]
        )
    for extra in ("gui", "rag", "lmstudio", "web", "mcp"):
        missing = missing_imports(extra)
        rows.append((f"deps:{extra}", "ok" if not missing else "missing: " + ", ".join(missing)))
    for key in cfg.doctor_env_keys:
        rows.append((f"env:{key}", os.environ.get(key, "")))
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    print(
        "\n".join(
            [
                f"llmmd {PROJECT_VERSION}",
                f"Project root: {cfg.paths.root}",
                f"Launcher config: {cfg.paths.config_file}",
                "",
                "Main commands:",
                "  python llmmd.py setup          Install deps, doctor, Qdrant Docker",
                "  python llmmd.py              Same as: python llmmd.py run",
                "  python llmmd.py run            Qdrant + RAG + панель с чатом",
            ]
        )
    )
    return 0


__all__ = [
    "cmd_doctor",
    "cmd_info",
    "cmd_lmstudio",
    "cmd_mcp",
    "cmd_mcp_config",
    "cmd_qdrant",
    "cmd_rag",
    "cmd_run",
    "cmd_runall",
    "cmd_setup",
    "cmd_setup_lite",
    "cmd_web",
]
