from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManagedProcess:
    name: str
    args: list[str]
    pid: int
    started_at: float
    log_path: Path
    popen: subprocess.Popen

    @property
    def is_running(self) -> bool:
        return self.popen.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.popen.poll()


def cli_command(root: Path, *args: str) -> list[str]:
    return [sys.executable, str(root / "llmmd.py"), *args]


def runtime_log_dir(root: Path) -> Path:
    path = root / ".runtime" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def start_cli_process(
    name: str,
    cli_args: list[str],
    *,
    root: Path,
    env: dict[str, str] | None = None,
) -> ManagedProcess:
    log_path = runtime_log_dir(root) / f"{name}.log"
    cmd = cli_command(root, *cli_args)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with log_path.open("ab") as log:
        header = f"\n\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(cmd)} ---\n".encode("utf-8")
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=full_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    return ManagedProcess(name=name, args=cli_args, pid=proc.pid, started_at=time.time(), log_path=log_path, popen=proc)


def stop_process(process: ManagedProcess, *, timeout_s: float = 10.0) -> None:
    if not process.is_running:
        return
    if os.name == "nt":
        process.popen.terminate()
    else:
        process.popen.send_signal(signal.SIGTERM)
    try:
        process.popen.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.popen.kill()
        process.popen.wait(timeout=timeout_s)


def tail_text(path: Path, *, lines: int = 120, max_bytes: int = 128_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
        except OSError:
            f.seek(0)
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def run_cli_capture(
    cli_args: list[str],
    *,
    root: Path,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cli_command(root, *cli_args),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
