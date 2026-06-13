# Agent Start Here

## Token-Saving Rules

- Do not scan `outputs/`, `.venv/`, `.pytest_cache/`, or `__pycache__/`.
- Start with `docs/REPO_MAP.md`.
- User-facing CLI: only `python llmmd.py setup` and `python llmmd.py` (run).
- Internal service commands (`rag`, `web`, `qdrant`, …) live in `llmmd_core/cli.py` → `INTERNAL_COMMANDS`.

## Fast Commands

```powershell
python llmmd.py setup
python llmmd.py
python -m pytest -q
```

## Architecture

`llmmd.py` → `llmmd_core/`. Qdrant: `python llmmd.py qdrant up` (internal, also used from setup/run).
