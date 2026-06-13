"""
Семантическая нарезка через OpenAI-совместимый API (LM Studio: обычно http://127.0.0.1:1234/v1).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import httpx

from .chunking import TextChunk, _split_oversized, split_markdown_by_heading_level
from .config import Settings

ChunkingMode = Literal["heading", "semantic", "heading_semantic"]


def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _parse_llm_chunks_json(content: str) -> list[tuple[str, str]]:
    s = _strip_json_fence(content)
    data = json.loads(s)
    arr = data.get("chunks")
    if not isinstance(arr, list):
        raise ValueError("JSON: ожидался объект с ключом chunks (массив)")
    out: list[tuple[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        h = str(item.get("heading", "") or "").strip() or "_semantic"
        t = str(item.get("text", "") or "").strip()
        if t:
            out.append((h, t))
    if not out:
        raise ValueError("Пустой массив chunks")
    return out


def _lm_chat(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    temperature: float,
) -> str:
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    url = root + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    try:
        return str(data["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Неожиданный ответ chat/completions: {data!r}") from e


_SYSTEM_PROMPT = """Ты разбиваешь технический markdown на семантические фрагменты для поиска (RAG).
Правила:
1) Верни ТОЛЬКО один JSON-объект без пояснений до и после.
2) Формат: {"chunks":[{"heading":"краткая тема на русском","text":"..."}]}
3) Поле text — это ДОСЛОВНЫЕ непрерывные подстроки из входного документа (копируй символ в символ фрагменты). Не перефразируй, не сокращай смысл, не выдумывай факты.
4) Разбей на столько чанков, сколько нужно по смене темы (часто 3–12; короткий текст — 1 чанк, длинный — до 24).
5) Заголовки markdown (# ##) по возможности не разрывай посередине: границу ставь между абзацами (\n\n)."""


def _split_windows(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = text.split("\n\n")
    windows: list[str] = []
    buf: list[str] = []
    cur = 0
    for p in parts:
        if len(p) > max_chars:
            if buf:
                windows.append("\n\n".join(buf))
                buf = []
                cur = 0
            for i in range(0, len(p), max_chars):
                windows.append(p[i : i + max_chars])
            continue
        add = len(p) + (2 if buf else 0)
        if cur + add > max_chars and buf:
            windows.append("\n\n".join(buf))
            buf = [p]
            cur = len(p)
        else:
            if buf:
                cur += 2
            buf.append(p)
            cur += len(p)
    if buf:
        windows.append("\n\n".join(buf))
    if not windows:
        return [text[:max_chars]]
    return windows


def _validate_substrings(full: str, chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Оставляем только чанки, реально встречающиеся в full (после нормализации переводов строк)."""
    if not full.strip():
        return []
    out: list[tuple[str, str]] = []
    for h, t in chunks:
        if t in full:
            out.append((h, t))
            continue
        t2 = t.replace("\r\n", "\n")
        f2 = full.replace("\r\n", "\n")
        if t2 in f2:
            out.append((h, t2))
    return out


def semantic_chunks_one_window(
    window: str,
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
    temperature: float,
    context_heading: str = "",
) -> list[tuple[str, str]]:
    ctx = f"Контекст раздела (родительский заголовок): {context_heading}\n\n" if context_heading else ""
    user = (
        ctx
        + "Разбей следующий документ на семантические чанки по правилам из system.\n\n"
        + "<document>\n"
        + window
        + "\n</document>"
    )
    raw = _lm_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        timeout=timeout,
        temperature=temperature,
    )
    parsed = _parse_llm_chunks_json(raw)
    good = _validate_substrings(window, parsed)
    if good:
        return good
    return [("_semantic_fallback", window.strip())]


def split_for_index(
    markdown: str,
    settings: Settings,
    chunking_mode: ChunkingMode,
    heading_level: int,
    chunk_max_chars: int,
    chunk_overlap_chars: int,
) -> list[TextChunk]:
    """
    heading — только скрипт по заголовкам;
    semantic — только LM Studio по окнам;
    heading_semantic — сначала заголовки, затем LM для слишком длинных секций.
    """
    if chunking_mode == "heading":
        return _heading_only_chunks(
            markdown,
            heading_level=heading_level,
            chunk_max_chars=chunk_max_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )

    base = settings.lm_studio_base_url.strip()
    model = settings.semantic_chunk_model.strip()
    if not model:
        raise ValueError("Задайте RAG_SEMANTIC_CHUNK_MODEL (имя модели в LM Studio)")

    if chunking_mode == "semantic":
        return _semantic_only_chunks(
            markdown,
            settings,
            chunk_max_chars=chunk_max_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )

    # heading_semantic
    sections = split_markdown_by_heading_level(markdown, level=heading_level)
    out: list[TextChunk] = []
    idx = 0
    min_sub = settings.semantic_subchunk_min_chars
    max_in = settings.semantic_chunk_max_input_chars
    for sec_h, content in sections:
        if not content.strip():
            continue
        blocks: list[tuple[str, str]] = []
        if len(content) < min_sub:
            blocks = [(sec_h, content)]
        else:
            wins = _split_windows(content, max_in)
            for w_i, win in enumerate(wins):
                ctx_h = sec_h if len(wins) == 1 else f"{sec_h} [LM окно {w_i + 1}]"
                sub = semantic_chunks_one_window(
                    win,
                    base_url=base,
                    api_key=settings.lm_studio_api_key,
                    model=model,
                    timeout=settings.semantic_llm_timeout_s,
                    temperature=settings.semantic_chunk_temperature,
                    context_heading=sec_h,
                )
                for sh, st in sub:
                    if sh == "_semantic_fallback":
                        blocks.append((ctx_h, st))
                    else:
                        blocks.append((f"{ctx_h} / {sh}", st))
        if chunk_max_chars > 0:
            merged: list[tuple[str, str]] = []
            for h, t in blocks:
                for piece in _split_oversized(t, chunk_max_chars, chunk_overlap_chars):
                    if piece.strip():
                        merged.append((h, piece.strip()))
            blocks = merged
        for h, t in blocks:
            if t.strip():
                out.append(TextChunk(text=t.strip(), heading=h, chunk_index=idx))
                idx += 1
    return out


def _heading_only_chunks(
    markdown: str,
    *,
    heading_level: int,
    chunk_max_chars: int,
    chunk_overlap_chars: int,
) -> list[TextChunk]:
    from .chunking import chunk_markdown_file

    return chunk_markdown_file(
        markdown,
        heading_level=heading_level,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )


def _semantic_only_chunks(
    markdown: str,
    settings: Settings,
    *,
    chunk_max_chars: int,
    chunk_overlap_chars: int,
) -> list[TextChunk]:
    base = settings.lm_studio_base_url.strip()
    model = settings.semantic_chunk_model.strip()
    max_in = settings.semantic_chunk_max_input_chars
    out: list[TextChunk] = []
    idx = 0
    wins = _split_windows(markdown, max_in)
    multi = len(wins) > 1
    for w_i, win in enumerate(wins):
        ctx = f"[Документ, часть {w_i + 1}]" if multi else ""
        pairs = semantic_chunks_one_window(
            win,
            base_url=base,
            api_key=settings.lm_studio_api_key,
            model=model,
            timeout=settings.semantic_llm_timeout_s,
            temperature=settings.semantic_chunk_temperature,
            context_heading=ctx,
        )
        merged: list[tuple[str, str]] = []
        for h, t in pairs:
            hh = f"{ctx} / {h}".strip(" /") if ctx else h
            if chunk_max_chars > 0:
                for piece in _split_oversized(t, chunk_max_chars, chunk_overlap_chars):
                    if piece.strip():
                        merged.append((hh, piece.strip()))
            else:
                if t.strip():
                    merged.append((hh, t.strip()))
        for hh, t in merged:
            out.append(TextChunk(text=t, heading=hh, chunk_index=idx))
            idx += 1
    return out


def normalize_chunking_mode(raw: str | None) -> ChunkingMode:
    s = (raw or "heading").strip().lower().replace("-", "_")
    if s in ("heading", "script", "markdown"):
        return "heading"
    if s in ("semantic", "llm", "lm_studio"):
        return "semantic"
    if s in ("heading_semantic", "hybrid", "heading_semantic_chunk"):
        return "heading_semantic"
    raise ValueError(
        f"Неизвестный chunking_mode={raw!r}; допустимо: heading, semantic, heading_semantic",
    )
