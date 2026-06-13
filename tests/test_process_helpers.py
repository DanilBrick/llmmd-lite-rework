from __future__ import annotations

import sys
from pathlib import Path

from llmmd_core.processes import cli_command, tail_text


def test_cli_command_targets_single_root_entrypoint():
    root = Path("C:/work/llmmd")

    assert cli_command(root, "rag") == [sys.executable, str(root / "llmmd.py"), "rag"]


def test_tail_text_reads_last_lines(tmp_path: Path):
    path = tmp_path / "app.log"
    path.write_text("\n".join(str(i) for i in range(10)), encoding="utf-8")

    assert tail_text(path, lines=3) == "7\n8\n9"
