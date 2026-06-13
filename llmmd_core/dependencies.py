from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .config import EXTRA_IMPORTS, VALID_EXTRAS


def missing_imports(extra: str) -> list[str]:
    return [name for name in EXTRA_IMPORTS.get(extra, ()) if importlib.util.find_spec(name) is None]


def run_pip_install(extra: str, *, root: Path) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extra}]"]
    subprocess.check_call(cmd, cwd=root)
    importlib.invalidate_caches()


def ensure_runtime(extra: str | Iterable[str], *, root: Path, no_install: bool = False) -> None:
    extras = [extra] if isinstance(extra, str) else list(extra)
    missing: dict[str, list[str]] = {}
    for item in extras:
        miss = missing_imports(item)
        if miss:
            missing[item] = miss
    if not missing:
        return

    details = "; ".join(f"{name}: {', '.join(mods)}" for name, mods in missing.items())
    if no_install or os.environ.get("LLMMD_SKIP_AUTO_INSTALL") == "1":
        raise RuntimeError(
            f"Missing Python dependencies ({details}). Run: "
            f"{sys.executable} -m pip install -e .[{','.join(extras)}]"
        )

    for item in missing:
        print(f"[llmmd] Installing missing dependency group '{item}' ({', '.join(missing[item])})...")
        run_pip_install(item, root=root)

    still_missing = {item: missing_imports(item) for item in extras}
    still_missing = {item: mods for item, mods in still_missing.items() if mods}
    if still_missing:
        details = "; ".join(f"{name}: {', '.join(mods)}" for name, mods in still_missing.items())
        raise RuntimeError(f"Dependencies are still missing after install: {details}")


def install_extras(extras: list[str], *, root: Path) -> None:
    selected = extras or ["all"]
    for extra in selected:
        if extra not in VALID_EXTRAS:
            raise RuntimeError(f"Unknown extra: {extra}")
        run_pip_install(extra, root=root)
