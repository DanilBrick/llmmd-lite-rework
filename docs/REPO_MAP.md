# Repository Map

Use this map before scanning source files.

## Entry Points

| Path | Role |
|---|---|
| `llmmd.py` | The only root Python entrypoint; delegates to `llmmd_core.cli`. |
| `llmmd_core/cli.py` | Argparse parser and global config loading. |
| `llmmd_core/commands.py` | User-facing command implementations (including `runall` composite). |

## Launcher Core

| Path | Role |
|---|---|
| `config/llmmd.yaml` | Editable launcher defaults: web/rag/qdrant/chat. |
| `llmmd_core/config.py` | Dataclasses and YAML loader for launcher config. |
| `llmmd_core/dependencies.py` | Dependency-group checks and `pip install -e .[extra]`. |
| `llmmd_core/docker.py` | Docker Compose lifecycle and Qdrant readiness probing. |

## OCR / GUI

| Path | Role |
|---|---|
| `gui/webui.py` | Streamlit lite UI: система, сервисы, RAG, OCR, чат, doctor. |
| `gui/services/file_processing.py` | Main conversion job, MarkItDown, splitting, LLM OCR orchestration. |
| `gui/services/pdf_images.py` | PDF image extraction and figure descriptions. |
| `gui/services/gui_settings.py` | GUI settings JSON store. |
| `gui/services/lmstudio_autoload_bridge.py` | OCR → lite LM Studio autoload bridge. |

## LM Studio Autoload (lite)

| Path | Role |
|---|---|
| `lmstudio_autoload/config.yaml` | Default model roles. |
| `lmstudio_autoload/service.py` | `ensure_model(role)` via LM Studio REST. |
| `lmstudio_autoload/api.py` | HTTP `/health`, `/ensure-model`. |

## MCP (lite)

| Path | Role |
|---|---|
| `claude_mcp/llmmd_rag_server.py` | stdio MCP: `rag_search`, `rag_ask`. |
| `llmmd_core/mcp.py` | MCP JSON for Cursor/Claude. |

## RAG

| Path | Role |
|---|---|
| `rag_service/main.py` | FastAPI app: health, settings, indexing, search, RAG. |
| `rag_service/config.py` | RAG settings from env/UI overlay. |
| `rag_service/ui_settings_store.py` | UI-persisted RAG settings. |
| `rag_service/store.py` | Qdrant schema validation, collection creation, upsert, search. |
| `rag_service/chunking.py` | Heading-based Markdown chunking. |
| `rag_service/semantic_chunking.py` | LM-assisted chunking. |
| `rag_service/embeddings.py` | Dense and optional sparse encoders. |
| `rag_service/runtime_status.py` | Service/indexing status snapshots. |
| `rag_service/docker-compose.qdrant.yml` | Local Qdrant service definition. |

## Tests

| Path | Covers |
|---|---|
| `tests/test_cli_launcher.py` | CLI parsing. |
| `tests/test_docker_launcher.py` | Docker/Qdrant command construction and config behavior. |
| `tests/test_launcher_config.py` | YAML launcher config parsing. |
| `tests/test_openwebui_compat.py` | Open WebUI adapter logic. |
| `tests/test_qdrant_schema.py` | Qdrant schema validation. |
| `tests/test_lmstudio_autoload.py` | Lite LM Studio autoload service. |
