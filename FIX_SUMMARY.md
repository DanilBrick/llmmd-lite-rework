# RAG DOCX → Markdown → Index → Chat Fix

## Problem
User tries to index `.docx` files directly, but system only indexes `.md` files.
Indexing completes in 0.2 seconds because no `.md` files are found.

## Solution

### 1. Convert DOCX to Markdown first
- Use UI: "📄 Обработка документов" → specify folder with `.docx` files
- System converts `.docx` → `.md` and saves to `outputs/` folder

### 2. Index the generated Markdown files
- Use UI: "🗂️ Индексация в БД" → specify `outputs/` folder
- System finds `.md` files and indexes them

### 3. Ask questions
- Use UI: "💬 Чат с документами" → ask questions about indexed documents

## Code Changes

### rag_service/docx_to_md.py (NEW)
- `convert_docx_to_markdown()` - single file conversion
- `convert_docx_folder_to_markdown()` - batch conversion
- `find_or_convert_docx_files()` - smart detection: use existing .md or convert .docx

### rag_service/chunking.py (MODIFIED)
- Added `find_or_convert_docx_files()` function
- Imports conversion functions from `docx_to_md.py`

### rag_service/main.py (MODIFIED)
- `IndexRequest.glob_pattern` changed from `"**/*.md"` to `"**/*.docx"`
- Index endpoint now calls `find_or_convert_docx_files()` instead of `iter_markdown_files()`

## Workflow for User

1. Place `.docx` files in: `C:\Users\Danil\llmmd-files-folder\input\`
2. Run UI: `python llmmd.py` → http://127.0.0.1:8501
3. "📄 Обработка документов" → path: `C:\Users\Danil\llmmd-files-folder\input\`
4. Click "Запустить OCR" → wait for conversion
5. "🗂️ Индексация в БД" → path: `C:\Users\Danil\llmmd-files-folder\outputs\`
6. Click "Начать индексацию" → wait for indexing
7. "💬 Чат с документами" → ask questions

## Expected Output
- `outputs/` folder contains `.md` files after step 4
- Indexing shows progress (not 0.2 seconds) after step 6
- RAG API returns results after step 7
