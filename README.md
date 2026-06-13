# llmmd lite

Локальный пайплайн: OCR → Markdown → RAG → чат в панели.

## Два шага

```powershell
python llmmd.py setup
python llmmd.py
```

`setup` — зависимости, проверка, Qdrant в Docker.

`python llmmd.py` — Qdrant, RAG API и веб-панель с чатом.

Панель: **http://127.0.0.1:8501** (блок «Чат с документами»).

## Опции

```powershell
python llmmd.py setup --skip-qdrant
python llmmd.py run --skip-qdrant
```

Ctrl+C останавливает RAG и панель; Qdrant в Docker остаётся запущенным.
