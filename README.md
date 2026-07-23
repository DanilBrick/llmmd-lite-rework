# llmmd lite Rework - Что это такое?

Это локальный пайплайн: OCR → Markdown → RAG → чат с LLM. Эта система позволяет обработать несколько файлов и получать ответы от языковой модели только на основе индексированных ею данных.

## Установка и запуск в два шага

Откройте терминал и введите команды:

```powershell
python llmmd.py setup
python llmmd.py
```

`setup` — зависимости, проверка, Qdrant в Docker.

`python llmmd.py` — запуск pipeline: Qdrant, RAG API и веб-панель с чатом.

Панель: **http://127.0.0.1:8501** (блок «Чат с документами»).

## Опции запуска

```powershell
python llmmd.py setup --skip-qdrant
python llmmd.py run --skip-qdrant
```

Ctrl+C останавливает RAG и панель; Qdrant в Docker остаётся запущенным.

## Файлы с инструкциями

Установка всех компонентов системы - в файле `INSTALL.md`