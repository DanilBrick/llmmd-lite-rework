# llmmd Architecture

`llmmd lite`: OCR → Markdown → RAG → чат в Streamlit-панели.

## Runtime

| Process | Command | Role |
|---|---|---|
| Web panel | `python llmmd.py` | Streamlit UI с RAG-чатом |
| Qdrant | internal `qdrant up` | Векторная БД (Docker) |
| RAG API | internal `rag` | Индексация и ответы |

## Quick Start

```powershell
python llmmd.py setup
python llmmd.py
```
