from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import QdrantDockerConfig


@dataclass(frozen=True)
class DockerStatus:
    ok: bool
    message: str


class _ComposeConfig(Protocol):
    compose_file: Path
    logs_tail: int


class _HttpServiceConfig(Protocol):
    url: str
    health_paths: tuple[str, ...]
    wait_timeout_s: float
    wait_interval_s: float


def docker_status(*, root: Path) -> DockerStatus:
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception as e:
        return DockerStatus(False, f"unavailable ({type(e).__name__}: {e})")
    if proc.returncode == 0:
        return DockerStatus(True, f"ok ({proc.stdout.strip()})")
    return DockerStatus(False, f"unavailable ({(proc.stderr or proc.stdout).strip()})")


def compose_command(config: _ComposeConfig, action: str, *args: str) -> list[str]:
    return ["docker", "compose", "-f", str(config.compose_file), action, *args]


def run_compose(
    config: _ComposeConfig,
    action: str,
    *,
    root: Path,
    extra_args: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
) -> int:
    if not config.compose_file.is_file():
        raise RuntimeError(f"Compose file not found: {config.compose_file}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.call(compose_command(config, action, *(extra_args or [])), cwd=root, env=env)


def _probe_urls(config: _HttpServiceConfig) -> list[str]:
    base = config.url.rstrip("/")
    return [base + path for path in config.health_paths]


def http_service_ready(config: _HttpServiceConfig, *, timeout_s: float = 2.0) -> tuple[bool, str]:
    errors: list[str] = []
    for url in _probe_urls(config):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                if 200 <= int(resp.status) < 500:
                    return True, f"ready ({url}, HTTP {resp.status})"
                errors.append(f"{url}: HTTP {resp.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
    return False, "; ".join(errors[-3:])


def _wait_for_http_service(config: _HttpServiceConfig, *, label: str) -> tuple[bool, str]:
    deadline = time.monotonic() + config.wait_timeout_s
    last = "not checked"
    while time.monotonic() < deadline:
        ok, message = http_service_ready(config, timeout_s=2.0)
        if ok:
            return True, message
        last = message
        time.sleep(config.wait_interval_s)
    return False, f"timeout after {config.wait_timeout_s:g}s; last probe: {last}"


def qdrant_probe_urls(config: QdrantDockerConfig) -> list[str]:
    return _probe_urls(config)


def qdrant_http_ready(config: QdrantDockerConfig, *, timeout_s: float = 2.0) -> tuple[bool, str]:
    return http_service_ready(config, timeout_s=timeout_s)


def wait_for_qdrant(config: QdrantDockerConfig) -> tuple[bool, str]:
    return _wait_for_http_service(config, label="Qdrant")


def qdrant_up(
    config: QdrantDockerConfig,
    *,
    root: Path,
    wait: bool = True,
    pull: bool = False,
) -> int:
    if pull:
        code = run_compose(config, "pull", root=root)
        if code:
            return code
    code = run_compose(config, "up", root=root, extra_args=["-d"])
    if code or not wait:
        return code
    ok, message = wait_for_qdrant(config)
    print(f"[llmmd] Qdrant {message}")
    return 0 if ok else 1


def qdrant_down(config: QdrantDockerConfig, *, root: Path, volumes: bool = False) -> int:
    args = ["-v"] if volumes else []
    return run_compose(config, "down", root=root, extra_args=args)


def qdrant_logs(config: QdrantDockerConfig, *, root: Path, follow: bool = False, tail: int | None = None) -> int:
    args = ["--tail", str(tail if tail is not None else config.logs_tail)]
    if follow:
        args.append("-f")
    return run_compose(config, "logs", root=root, extra_args=args)


def qdrant_ps(config: QdrantDockerConfig, *, root: Path) -> int:
    return run_compose(config, "ps", root=root)


def qdrant_status(config: QdrantDockerConfig, *, root: Path) -> int:
    ds = docker_status(root=root)
    print(f"docker: {ds.message}")
    ok, message = qdrant_http_ready(config, timeout_s=0.75)
    print(f"qdrant: {message}")
    return 0 if ds.ok and ok else 1
