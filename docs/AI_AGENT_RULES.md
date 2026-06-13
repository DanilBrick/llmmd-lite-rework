# AI Agent Rules

Follow these boundaries before adding files or searching broadly.

## Search Budget

1. Read `AGENTS.md`.
2. Read `docs/REPO_MAP.md`.
3. Run targeted `rg` only inside the likely module.

Default file discovery:

```powershell
rg --files -g '!outputs/**' -g '!.venv/**' -g '!__pycache__/**'
```

Avoid opening `outputs/` (generated data) and `.venv/`.

## Reuse Rules

| Responsibility | Owner |
|---|---|
| CLI, process launch | `llmmd_core/cli.py`, `llmmd_core/commands.py` |
| Launcher defaults | `config/llmmd.yaml`, `llmmd_core/config.py` |
| Docker/Qdrant lifecycle | `llmmd_core/docker.py` |
| Qdrant schema/search | `rag_service/store.py` |
| RAG API | `rag_service/main.py` |
| RAG UI settings | `rag_service/ui_settings_store.py` |
| OCR settings | `gui/services/gui_settings.py` |
| OCR conversion job | `gui/services/file_processing.py` |

## Test Rules

```powershell
python -m pytest -q
```
