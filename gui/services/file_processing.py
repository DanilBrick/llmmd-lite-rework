"""Парсинг PDF, очистка markdown, разбиение на секции и конвертация через MarkItDown."""

from __future__ import annotations

import io
import json
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

try:
    from markitdown import MarkItDown, StreamInfo
except Exception:
    MarkItDown = None
    StreamInfo = None

from .lmstudio_autoload_bridge import ensure_lmstudio_roles, resolve_llm_models
from .ocr_runtime import (
    install_ocr_gui_logging,
    install_ocr_image_limits,
    install_pdf_cancel_hooks,
    set_pdf_cancel_event,
    set_pdf_pause_event,
)
from .pdf_images import (
    build_figures_markdown_section,
    collect_figure_descriptions,
    extract_figures_from_pdf,
)
from .pdf_raster_profile import (
    PdfRasterProfile,
    probe_pdf_raster_profile,
    set_active_pdf_raster_profile,
)
from .duration_fmt import format_duration_ru

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


PIPELINE_STAGE_DESCRIPTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "convert",
        "title": "OCR / конвертация",
        "description": "Обязательный этап: MarkItDown читает файл и получает первичный Markdown.",
    },
    {
        "id": "formulas",
        "title": "Формулы -> LaTeX",
        "description": "Опциональный LLM-проход по фрагментам с формулами; можно выключать для книг без математики.",
    },
    {
        "id": "title",
        "title": "ИИ-название книги",
        "description": "Один текстовый запрос пытается взять официальное название с титула; при выключении используются metadata/filename.",
    },
    {
        "id": "images_extract",
        "title": "Извлечение картинок",
        "description": "Сохраняет встроенные крупные изображения PDF в папку {имя}_assets рядом с .md.",
    },
    {
        "id": "images_describe",
        "title": "Описание картинок",
        "description": "Vision-модель кратко описывает извлечённые рисунки; самый дорогой этап по времени.",
    },
    {
        "id": "split",
        "title": "Разбиение на главы",
        "description": "Делит Markdown на подфайлы по заголовкам или LLM-якорям.",
    },
    {
        "id": "write_combined",
        "title": "Общий Markdown",
        "description": "Сохраняет полный Markdown-файл рядом с папками глав и картинок.",
    },
)


def default_enabled_stages() -> list[str]:
    return [item["id"] for item in PIPELINE_STAGE_DESCRIPTIONS]


def normalize_enabled_stages(value: Any) -> set[str]:
    known = set(default_enabled_stages())
    if value is None:
        return known
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw = [str(part).strip() for part in value]
    else:
        return known
    selected = {part for part in raw if part in known}
    return selected or known


def parse_pdf_page_spec(spec: str) -> Optional[tuple[int, ...]]:
    s = (spec or "").strip()
    if not s:
        return None
    pages: set[int] = set()
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    ordered = tuple(sorted(pages))
    if any(p < 1 for p in ordered):
        raise ValueError("Номера страниц должны быть не меньше 1")
    return ordered


def pdf_pages_filename_suffix(pages: tuple[int, ...]) -> str:
    if len(pages) <= 6:
        return "_p" + "-".join(str(p) for p in pages)
    return f"_p{pages[0]}-{pages[-1]}_{len(pages)}стр"


def sanitize_output_stem(name: str, max_len: int = 120) -> str:
    """Безопасное имя для .md / папок: без слэшей и управляющих символов, обрезка длины."""
    name = (name or "").strip()
    if not name:
        return ""
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"[\x00-\x1f\x7f]+", "", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")
    if not name or name in (".", ".."):
        return ""
    return name[:max_len]


_GARBAGE_PDF_TITLES = frozenset(
    {
        "untitled",
        "unknown",
        "title",
        "document",
        "pdf",
        "pdf document",
        "no title",
        "microsoft word",
    }
)


def pdf_title_from_metadata(pdf_path: Path) -> str:
    """Заголовок из PDF metadata (Title), очищенный; пустая строка если не годится."""
    import fitz

    try:
        doc = fitz.open(str(pdf_path))
        try:
            raw = (doc.metadata or {}).get("title") or ""
        finally:
            doc.close()
    except Exception:
        return ""
    t = (raw or "").strip()
    if len(t) < 2:
        return ""
    low = re.sub(r"\s+", " ", t.lower())
    if low in _GARBAGE_PDF_TITLES or low.startswith("microsoft word "):
        return ""
    return sanitize_output_stem(t)


def resolve_output_base_stem(
    src: Path,
    stem_extra: str,
    manual_stem: str,
    prefer_pdf_meta: bool,
) -> tuple[str, str]:
    """
    Имя без расширения для выходных .md, папок сплита и _assets.
    Возвращает (base_stem, краткая подпись источника для лога).
    """
    manual = sanitize_output_stem(manual_stem)
    if manual:
        return manual + stem_extra, "вручную"
    if src.suffix.lower() == ".pdf" and prefer_pdf_meta:
        meta = pdf_title_from_metadata(src)
        if meta:
            return meta + stem_extra, "Title в PDF"
    return src.stem + stem_extra, "имя файла"


def unique_output_stem(out_dir: Path, desired_stem: str) -> str:
    """Единый stem для .md, папки сплита, assets и отчёта без перезаписи прошлых артефактов."""
    base = sanitize_output_stem(desired_stem) or "document"

    def taken(stem: str) -> bool:
        return any(
            (
                (out_dir / f"{stem}.md").exists(),
                (out_dir / stem).exists(),
                (out_dir / f"{stem}_assets").exists(),
                (out_dir / f"{stem}_figures.md").exists(),
                (out_dir / f"{stem}_report.md").exists(),
            )
        )

    if not taken(base):
        return base
    suffix = 1
    while taken(f"{base}_{suffix}"):
        suffix += 1
    return f"{base}_{suffix}"


def build_pdf_subset_bytes(path: Path, pages: tuple[int, ...]) -> bytes:
    import fitz
    doc = fitz.open(str(path))
    try:
        n = doc.page_count
        for p in pages:
            if p > n:
                raise ValueError(f"Страница {p} вне диапазона (в документе {n} стр.)")
        out = fitz.open()
        try:
            for p in pages:
                out.insert_pdf(doc, from_page=p - 1, to_page=p - 1)
            return out.tobytes()
        finally:
            out.close()
    finally:
        doc.close()


_CID_TOKEN_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)


def _line_is_pdf_cid_garbage(line: str) -> bool:
    s = line.strip()
    if not s or "(cid:" not in s.lower():
        return False
    matches = list(_CID_TOKEN_RE.finditer(s))
    if len(matches) < 2:
        return False
    non_space = len(re.sub(r"\s", "", s))
    if non_space < 1:
        return False
    cid_chars = sum(len(m.group(0)) for m in matches)
    ratio = cid_chars / non_space
    if ratio >= 0.38 and len(matches) >= 4:
        return True
    if ratio >= 0.50 and len(matches) >= 3:
        return True
    if ratio >= 0.62 and len(matches) >= 2:
        return True
    return False


def strip_pdf_cid_garbage(md: str) -> tuple[str, int]:
    lines = md.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if _line_is_pdf_cid_garbage(line):
            removed += 1
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    return out, removed


SPLIT_LEVEL_OPTIONS = ["# (H1)", "## (H2)", "### (H3)"]
SPLIT_LEVEL_MAP = {"# (H1)": 1, "## (H2)": 2, "### (H3)": 3}

LLM_TOC_CHUNK_MAX_CHARS = 7_600
LLM_TOC_CHUNK_OVERLAP = 1_200

# Под локальные серверы с n_ctx ~12k: чанк + системный промпт + шаблон чата + max_tokens не должны упираться в слот.
LLM_FORMULA_MAX_CHUNK = 5_600
LLM_TOC_MAX_COMPLETION_TOKENS = 2048
LLM_FORMULA_MAX_COMPLETION_CAP = 3584

_LLM_THINKING_BLOCK_RE = re.compile(
    r"(?is)<(?:think|redacted_thinking|reasoning)[^>]*>.*?</(?:think|redacted_thinking|reasoning)>\s*"
)
_LLM_THINKING_OPEN_RE = re.compile(
    r"(?is)^\s*<(?:think|redacted_thinking|reasoning)[^>]*>.*$"
)


def _llm_model_likely_thinking(model: str) -> bool:
    m = (model or "").lower()
    return "qwen" in m


def _strip_llm_thinking_blocks(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    for _ in range(8):
        prev = s
        s = _LLM_THINKING_BLOCK_RE.sub("", s).strip()
        if s == prev:
            break
    if _LLM_THINKING_OPEN_RE.match(s):
        return ""
    return s


def _openai_assistant_text(message: Any) -> str:
    """Текст ответа ассистента; для Qwen/LM Studio — content + reasoning_content, без think-блоков."""
    parts: list[str] = []
    for attr in ("content", "reasoning_content"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    combined = "\n".join(parts)
    stripped = _strip_llm_thinking_blocks(combined)
    return stripped.strip() or combined.strip()


def _llm_json_chat_completion(
    client: Any,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    create_kwargs: dict[str, Any] = {
        "model": model or "gpt-4o",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if _llm_model_likely_thinking(model):
        messages.append({"role": "assistant", "content": "\n\n\n\n"})
        create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(**create_kwargs)
    return _openai_assistant_text(resp.choices[0].message)


FORMULA_LATEX_SYSTEM = """Ты редактор технического markdown после OCR (PDF/DOCX → текст).

Задача: в переданном фрагменте markdown найди математические выражения, записанные «как в книге»
(дроби через / или горизонтальную черту, индексы/степени юникодом, символы Σ∫√αβ и т.п., смесь букв и знаков = < ≤ ≥ ±)
и перепиши их в корректный LaTeX для рендеринга (MathJax / KaTeX).

Правила:
— Строки-формулы и выражения в тексте: инлайн оборачивай в $...$; многострочные или крупные блоки — в $$...$$ (отдельные строки).
— Уже корректный LaTeX в $...$ / $$...$$ / \\( ... \\) не меняй без необходимости.
— Не меняй обычный связный текст, заголовки markdown (#), списки, таблицы, ссылки, картинки ![], блоки кода ```, HTML-комментарии <!-- -->.
— Не выдумывай новых формул: только то, что явно выглядит как формула или обозначение в исходнике.
— Язык документа (русский и т.д.) сохраняй; переводить текст не нужно.

Ответ: ОДИН блок кода markdown — три обратные кавычки, затем необязательно markdown, перевод строки, затем ПОЛНЫЙ преобразованный фрагмент
(тот же объём смысла, что во входе), затем перевод строки и закрывающие три кавычки. Без пояснений до и после блока."""


LLM_CONTEXT_SPLIT_MAX_DEPTH = 16
LLM_CONTEXT_SPLIT_MIN_CHARS = 280


def _stage_event(
    events: queue.Queue,
    *,
    stage_id: str,
    status: str,
    stage: str,
    label: str,
    file_path: str | None = None,
    file_index: int | None = None,
    files_total: int | None = None,
    model: str | None = None,
    details: str = "",
    started_at_unix: float | None = None,
    duration_s: float | None = None,
) -> None:
    events.put(
        (
            "stage",
            {
                "stage_id": stage_id,
                "status": status,
                "stage": stage,
                "label": label,
                "file_path": file_path,
                "file_name": Path(file_path).name if file_path else "",
                "file_index": file_index,
                "files_total": files_total,
                "model": model or "",
                "details": details,
                "started_at_unix": started_at_unix,
                "duration_s": duration_s,
            },
        )
    )


def _is_context_overflow_error(exc: BaseException) -> bool:
    """Срабатывает для llama.cpp LM Studio / подобных сообщений о привышении n_ctx."""
    parts: list[str] = [str(exc).lower()]
    msg = getattr(exc, "message", None)
    if msg:
        parts.append(str(msg).lower())
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            parts.append(str(err["message"]).lower())
        elif isinstance(err, str):
            parts.append(err.lower())
    blob = " ".join(parts)
    markers = (
        "exceeds the available context",
        "context window",
        "context length",
        "maximum context",
        "requested exceeds",
        "token limit",
        "maximum number of tokens",
        "too many tokens",
        "prompt is too long",
        "n_ctx",
        "maximum prompt",
    )
    return any(m in blob for m in markers)


def _bisect_text_for_llm_retry(text: str, min_chars: int = LLM_CONTEXT_SPLIT_MIN_CHARS) -> Optional[tuple[str, str]]:
    """Делит текст пополам по возможности на границе абзаца; None если делить нельзя."""
    s = text or ""
    n = len(s)
    if n < 2 * min_chars:
        return None
    mid = n // 2
    half_span = min(n // 3, max(480, min_chars * 4))
    lo = max(min_chars, mid - half_span)
    hi = min(n - min_chars, mid + half_span)
    window = s[lo:hi]
    cut = mid
    j_pp = window.rfind("\n\n")
    rel = lo + j_pp + 2 if j_pp >= 0 else -1
    if rel >= min_chars and rel <= n - min_chars:
        cut = rel
    else:
        j_nl = window.rfind("\n")
        rel2 = lo + j_nl + 1 if j_nl >= 0 else -1
        if rel2 >= min_chars and rel2 <= n - min_chars:
            cut = rel2
    a, b = s[:cut], s[cut:]
    if len(a.strip()) < min_chars // 2 or len(b.strip()) < min_chars // 2:
        return None
    return a, b


def _llm_formula_max_out_tokens(chunk_len: int) -> int:
    return min(LLM_FORMULA_MAX_COMPLETION_CAP, max(512, int(chunk_len / 2.2)))


def _llm_formula_chat_once(client: Any, model: str, chunk: str) -> str:
    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[
            {"role": "system", "content": FORMULA_LATEX_SYSTEM},
            {
                "role": "user",
                "content": "Фрагмент markdown для правки формул (верни его целиком в одном fenced-блоке):\n\n"
                + chunk,
            },
        ],
        temperature=0.1,
        max_tokens=_llm_formula_max_out_tokens(len(chunk)),
    )
    return (resp.choices[0].message.content or "").strip()


def _llm_convert_formula_segment(
    client: Any,
    model: str,
    chunk: str,
    events_queue: Optional[queue.Queue],
    split_depth: int,
) -> str:
    """Один сегмент формул; при переполнении контекста делит пополам и склеивает."""
    if not (chunk or "").strip():
        return chunk
    try:
        raw = _llm_formula_chat_once(client, model, chunk)
        converted = _parse_llm_formula_markdown_response(raw)
        if not converted.strip():
            return chunk
        if len(chunk) > 400 and len(converted) < len(chunk) * 0.25:
            return chunk
        return converted
    except Exception as e:
        if not _is_context_overflow_error(e) or split_depth >= LLM_CONTEXT_SPLIT_MAX_DEPTH:
            raise
        halves = _bisect_text_for_llm_retry(chunk)
        if not halves:
            raise
        a, b = halves
        if events_queue:
            events_queue.put(
                (
                    "log",
                    f"  формулы → LaTeX: лимит контекста ({split_depth + 1}× деление) "
                    f"— кусок {len(chunk):,} симв. → {len(a):,} + {len(b):,} симв.",
                )
            )
        left = _llm_convert_formula_segment(client, model, a, events_queue, split_depth + 1)
        right = _llm_convert_formula_segment(client, model, b, events_queue, split_depth + 1)
        return left + right


def _chunk_text_for_formula_llm(text: str, max_chars: int) -> list[str]:
    """Нарезка по границам абзацев, чтобы не резать формулы посередине."""
    n = len(text)
    if n <= max_chars:
        return [text]
    out: list[str] = []
    i = 0
    while i < n:
        remain = n - i
        if remain <= max_chars:
            out.append(text[i:])
            break
        window = text[i : i + max_chars]
        j = window.rfind("\n\n")
        if j >= max_chars // 5:
            end = i + j + 2
        else:
            j2 = window.rfind("\n")
            if j2 >= max_chars // 6:
                end = i + j2 + 1
            else:
                end = i + max_chars
        if end <= i:
            end = min(i + max_chars, n)
        out.append(text[i:end])
        i = end
    return out


_FORMULA_FENCE_RE = re.compile(r"```(?:markdown|md)?\s*\n([\s\S]*?)```", re.IGNORECASE)


def _parse_llm_formula_markdown_response(raw: str) -> str:
    """Достаёт markdown из ответа модели (ограждение ```markdown ... ```)."""
    s = (raw or "").strip()
    if not s:
        return ""
    m = _FORMULA_FENCE_RE.search(s)
    if m:
        return m.group(1).strip("\n")
    if s.startswith("```"):
        s2 = re.sub(r"^```(?:markdown|md)?\s*\n?", "", s, flags=re.IGNORECASE)
        s2 = re.sub(r"\n```\s*$", "", s2)
        return s2.strip("\n")
    if s.startswith("{") and '"markdown"' in s[:120]:
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            lo, hi = s.find("{"), s.rfind("}")
            if lo != -1 and hi > lo:
                try:
                    data = json.loads(s[lo : hi + 1])
                except json.JSONDecodeError:
                    return s
            else:
                return s
        if isinstance(data, dict):
            md = data.get("markdown")
            if isinstance(md, str):
                return md.strip("\n")
    return s


def llm_convert_formulas_in_markdown(
    client: Any,
    model: str,
    text: str,
    events_queue: Optional[queue.Queue] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """
    Проходит markdown и через LLM оборачивает/переписывает формулы в LaTeX ($...$, $$...$$).
    Длинный текст нарезается по абзацам; при ошибке на куске сохраняется исходный кусок.
    """
    if not (text or "").strip():
        return text
    chunks = _chunk_text_for_formula_llm(text, LLM_FORMULA_MAX_CHUNK)
    if len(chunks) > 1 and events_queue:
        events_queue.put(
            (
                "log",
                f"  формулы → LaTeX (LLM): документ разбит на {len(chunks)} частей (~{LLM_FORMULA_MAX_CHUNK} симв.).",
            )
        )
    parts: list[str] = []
    for ci, chunk in enumerate(chunks, start=1):
        if cancel_event is not None and cancel_event.is_set():
            if events_queue:
                events_queue.put(
                    ("log", "  формулы → LaTeX (LLM): остановка — хвост без преобразования.")
                )
            parts.extend(chunks[ci - 1 :])
            break
        if not chunk:
            continue
        try:
            if events_queue and len(chunks) > 1:
                events_queue.put(("log", f"  формулы → LaTeX: часть {ci}/{len(chunks)} ({len(chunk):,} симв.)…"))
            converted = _llm_convert_formula_segment(client, model, chunk, events_queue, 0)
            parts.append(converted)
        except Exception as e:
            if events_queue:
                events_queue.put(("log", f"  формулы → LaTeX: часть {ci} — {e}; оставлен исходник."))
            parts.append(chunk)
    return "".join(parts)


TOC_JSON_SYSTEM = """Ты помощник по структурированию текста после OCR (PDF → markdown).
Ответь ТОЛЬКО одним JSON-объектом без markdown-обёрток, без комментариев.

Документ иногда передаётся фрагментом: включай только разделы, чьё начало есть в этом фрагменте; якорь — дословная цитата из того же текста ниже.

Формат (предпочтительно): {"sections": [{"anchor": "...", "title": "..."}, ...]}
Либо кратко: {"sections": ["якорь1", "якорь2"]} — тогда title = короткая форма якоря.

Поле anchor — подстрока из переданного текста: начало раздела/главы/крупного блока (часто первая строка абзаца-заголовка).
anchor должна находиться в тексте посимвольно или с теми же переносами строк; копируй из текста, не перефразируй.

Поле title — короткое человекочитаемое имя раздела для оглавления (без «Рисунок», без номеров страниц).

Правила отбора:
— НЕ начинай раздел с подписей к рисункам/таблицам («Рисунок 2.5», «Таблица 1»), с колонтитулов, номеров страниц, строк «Page 31».
— НЕ дроби один абзац на несколько «глав»; якорь — начало смыслового блока (Глава N, Раздел, приложение, крупный подзаголовок документа).
— Порядок sections — сверху вниз. Без дубликатов якорей. Разумно 2–80 пунктов.
Если структуры нет — {"sections": []}."""


def _short_section_title(anchor: str, max_len: int = 72) -> str:
    line = (anchor or "").split("\n", 1)[0].strip()
    line = re.sub(r"\s+", " ", line)
    if len(line) > max_len:
        return line[: max_len - 1].rstrip() + "…"
    return line


def _parse_llm_toc_json(raw: str) -> list[tuple[str, str]]:
    """Возвращает список (anchor, title) для нарезки; title — подпись файла и ##."""
    s = (raw or "").strip()
    if not s:
        return []
    candidates = [s]
    stripped = _strip_llm_thinking_blocks(s)
    if stripped and stripped != s:
        candidates.append(stripped)
    for candidate in candidates:
        items = _parse_llm_toc_json_candidate(candidate)
        if items:
            return items
    return _parse_llm_toc_json_candidate(candidates[-1])


def _parse_llm_toc_json_candidate(s: str) -> list[tuple[str, str]]:
    s = (s or "").strip()
    if not s:
        return []
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        lo, hi = s.find("{"), s.rfind("}")
        if lo == -1 or hi <= lo:
            return []
        try:
            data = json.loads(s[lo : hi + 1])
        except json.JSONDecodeError:
            return []
    sec = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(sec, list):
        return []
    out: list[tuple[str, str]] = []
    seen_anchors: set[str] = set()
    for x in sec:
        anchor = ""
        title = ""
        if isinstance(x, str):
            anchor = x.strip()
            title = _short_section_title(anchor)
        elif isinstance(x, dict):
            anchor = (
                str(x.get("anchor") or x.get("beginning") or x.get("start") or "").strip()
            )
            title = str(x.get("title") or x.get("name") or x.get("heading") or "").strip()
            if anchor and not title:
                title = _short_section_title(anchor)
            elif title and not anchor:
                anchor = title
        if not anchor:
            continue
        if anchor in seen_anchors:
            continue
        seen_anchors.add(anchor)
        out.append((anchor, title or _short_section_title(anchor)))
    return out


def _chunks_for_llm_toc(text: str, max_chars: int, overlap: int) -> list[tuple[int, str]]:
    """Фрагменты с наложением: (абсолютный offset, подстрока из полного текста для prompt)."""
    n = len(text)
    mc = max(4096, int(max_chars))
    ov = max(0, min(int(overlap), mc // 2))
    if n <= mc:
        return [(0, text)]

    out: list[tuple[int, str]] = []
    start = 0
    hops = 0
    hop_limit = max(96, min(8192, n // max(mc - ov or mc // 4, 1) + 32))

    while start < n and hops < hop_limit:
        hops += 1
        hard_end = min(n, start + mc)
        cut = hard_end
        if hard_end < n:
            window = text[start:hard_end]
            j_pp = window.rfind("\n\n")
            if j_pp >= max(len(window) // 7, mc // 24):
                cut = start + j_pp + 2
            else:
                j_nl = window.rfind("\n")
                if j_nl >= max(len(window) // 8, mc // 32):
                    cut = start + j_nl + 1
        slice_ = text[start:cut]
        if slice_.strip():
            out.append((start, slice_))
        if cut >= n:
            break
        advance = cut - ov
        if advance <= start:
            advance = cut
        next_start = max(start + max(512, mc // 100), advance)
        if next_start >= n:
            break
        if next_start == start:
            next_start += max(256, mc // 128)
        start = next_start

    return out if out else [(0, text)]


def _merge_toc_json_batches(full_text: str, batches: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Якори из нескольких LLM-ответов по фрагментам; без дубликатов, порядок по позиции в документе."""
    by_anchor: dict[str, str] = {}
    insert_order = 0
    order_index: dict[str, int] = {}
    for batch in batches:
        for anchor, title in batch:
            a = (anchor or "").strip()
            if not a:
                continue
            t = (title or _short_section_title(a)).strip() or _short_section_title(a)
            if a not in by_anchor:
                by_anchor[a] = t
                order_index[a] = insert_order
                insert_order += 1

    items = list(by_anchor.items())

    def sort_key(kv: tuple[str, str]) -> tuple[int, int]:
        anchor, _ = kv
        pos = _find_toc_anchor(full_text, anchor, 0)
        if pos < 0:
            return (2_147_483_000 + order_index.get(anchor, insert_order), 0)
        return (pos, order_index.get(anchor, insert_order))

    items.sort(key=sort_key)
    return [(a, t) for a, t in items]


def _llm_toc_parse_one_response(
    client: Any,
    model: str,
    user_content: str,
    events_queue: Optional[queue.Queue] = None,
) -> list[tuple[str, str]]:
    raw = _llm_json_chat_completion(
        client,
        model,
        TOC_JSON_SYSTEM,
        user_content,
        temperature=0.15,
        max_tokens=LLM_TOC_MAX_COMPLETION_TOKENS,
    )
    items = _parse_llm_toc_json(raw)
    if events_queue and not items and (raw or "").strip():
        snippet = re.sub(r"\s+", " ", raw)[:140]
        events_queue.put(("log", f"  LLM TOC: JSON не распознан: «{snippet}»"))
    return items


def _llm_toc_items_with_context_split(
    client: Any,
    model: str,
    frag_idx: int,
    total_fr: int,
    abs_start: int,
    doc_len: int,
    body: str,
    events_queue: Optional[queue.Queue],
    cancel_event: Optional[threading.Event],
    split_depth: int = 0,
) -> list[tuple[str, str]]:
    if cancel_event is not None and cancel_event.is_set():
        return []
    sub_note = ""
    if split_depth:
        sub_note = (
            "Сервер отклонил слишком длинный промпт — подзадача после автоматического деления; "
            "те же правила JSON, якорь только из текста ниже.\n\n"
        )
    user = sub_note + (
        "Проанализируй текст ниже и верни JSON: sections — массив объектов "
        '{"anchor": "<дословный префикс из переданного фрагмента с места начала раздела>", '
        '"title": "<краткое имя раздела>"}.\n'
        "Включай только разделы, которые НАЧИНАЮТСЯ внутри этого фрагмента; якорь копируй из фрагмента.\n\n"
        f"Фрагмент {frag_idx} из {total_fr}; смещение в полном тексте ~{abs_start:,}, "
        f"длина {len(body):,} симв. всего в документе ~{doc_len:,}.\n\n"
        + body
    )
    try:
        return _llm_toc_parse_one_response(client, model, user, events_queue)
    except Exception as e:
        if cancel_event is not None and cancel_event.is_set():
            return []
        if not _is_context_overflow_error(e) or split_depth >= LLM_CONTEXT_SPLIT_MAX_DEPTH:
            raise
        halves = _bisect_text_for_llm_retry(body)
        if not halves:
            raise
        a, b = halves
        if events_queue:
            events_queue.put(
                (
                    "log",
                    f"  LLM TOC: лимит контекста — делю фрагмент {frag_idx}/{total_fr} "
                    f"(~{len(body):,} симв.) → ~{len(a):,} + ~{len(b):,} симв.",
                )
            )
        left = _llm_toc_items_with_context_split(
            client,
            model,
            frag_idx,
            total_fr,
            abs_start,
            doc_len,
            a,
            events_queue,
            cancel_event,
            split_depth + 1,
        )
        right = _llm_toc_items_with_context_split(
            client,
            model,
            frag_idx,
            total_fr,
            abs_start + len(a),
            doc_len,
            b,
            events_queue,
            cancel_event,
            split_depth + 1,
        )
        return left + right


def llm_infer_toc_sections(
    client: Any,
    model: str,
    text: str,
    events_queue: Optional[queue.Queue] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[tuple[str, str]]:
    raw_text = text or ""
    n = len(raw_text)
    if not raw_text.strip():
        return []

    if n <= LLM_TOC_CHUNK_MAX_CHARS:
        fragments = [(0, raw_text)]
    else:
        fragments = _chunks_for_llm_toc(raw_text, LLM_TOC_CHUNK_MAX_CHARS, LLM_TOC_CHUNK_OVERLAP)
        if events_queue:
            events_queue.put(
                (
                    "log",
                    "  LLM TOC: текст длинный — несколько последовательных запросов "
                    f"({LLM_TOC_CHUNK_MAX_CHARS:,}+/{LLM_TOC_CHUNK_OVERLAP:,} симв. налож.).",
                )
            )

    total_fr = len(fragments)
    batches: list[list[tuple[str, str]]] = []

    frag_no = 0
    for abs_start, body in fragments:
        frag_no += 1
        if cancel_event is not None and cancel_event.is_set():
            if events_queue:
                events_queue.put(
                    ("log", f"  LLM TOC: остановлено пользователем после части {frag_no}/{total_fr}.")
                )
            break
        if not (body or "").strip():
            continue
        if events_queue and total_fr > 1:
            events_queue.put(
                (
                    "log",
                    f"  LLM TOC: часть {frag_no}/{total_fr}: в промпт ~{len(body):,} симв. (~{abs_start:,}…).",
                )
            )
        elif events_queue:
            events_queue.put(("log", f"  LLM TOC: в промпт ~{len(body):,} симв.; компактный JSON-ответ."))
        try:
            items = _llm_toc_items_with_context_split(
                client,
                model,
                frag_no,
                total_fr,
                abs_start,
                n,
                body,
                events_queue,
                cancel_event,
                0,
            )
            batches.append(items)
            if events_queue and total_fr > 1:
                events_queue.put(
                    ("log", f"  LLM TOC: часть {frag_no}/{total_fr} — распознано пунктов JSON: {len(items)}.")
                )
        except Exception as e:
            if events_queue:
                events_queue.put(("log", f"  LLM TOC: часть {frag_no}/{total_fr} — {e}"))
            batches.append([])

    merged = _merge_toc_json_batches(raw_text, batches)
    if events_queue:
        events_queue.put(
            (
                "log",
                f"  LLM TOC: суммарно уникальных якорей из LLM: {len(merged)}.",
            )
        )
    return merged


def infer_toc_sections_combined(
    client: Any,
    model: str,
    text: str,
    events_queue: Optional[queue.Queue] = None,
    cancel_event: Optional[threading.Event] = None,
    *,
    use_llm: bool = True,
    heuristic_min_to_skip_llm: int = 2,
) -> list[tuple[str, str]]:
    """Эвристика по тексту + опционально LLM; объединённые якоря для нарезки."""
    raw_text = text or ""
    heuristic = infer_toc_sections_heuristic(raw_text, events_queue)
    if not use_llm:
        return heuristic
    if len(heuristic) >= heuristic_min_to_skip_llm:
        if events_queue:
            events_queue.put(
                (
                    "log",
                    f"  TOC: эвристика нашла {len(heuristic)} якорей — LLM TOC пропущен "
                    f"(порог {heuristic_min_to_skip_llm}).",
                )
            )
        return heuristic
    llm_items = llm_infer_toc_sections(
        client, model, raw_text, events_queue, cancel_event
    )
    combined = _merge_toc_json_batches(raw_text, [heuristic, llm_items])
    if events_queue:
        events_queue.put(
            (
                "log",
                f"  TOC: итого якорей (эвристика {len(heuristic)} + LLM {len(llm_items)} "
                f"→ уникальных {len(combined)}).",
            )
        )
    return combined


LLM_BOOK_TITLE_MAX_INPUT_CHARS = 6_500
LLM_BOOK_TITLE_FOCUS_CHARS = 2_500
LLM_BOOK_TITLE_MAX_COMPLETION_TOKENS = 512

BOOK_TITLE_JSON_SYSTEM = """Ты извлекаешь официальное название издания для имени файла после конвертации в markdown.
Ответь ТОЛЬКО одним JSON без markdown-обёрток: {"title": "<строка>"}.

Поле title:
— 2–90 символов, как на титульном листе, обложке или в поле Title (без расширения .md);
— язык title — как у документа;
— только шапка издания: книга, стандарт (ГОСТ/ISO и т.п.), методичка, отчёт, сборник;
— без символов \\ / : * ? \" < > | ;
— без кавычек внутри title.

Запрещено в title:
— пересказ содержания, тема главы, «как авторы…», «данная книга описывает…»;
— слова «фрагмент», «документ о», «пособие по теме», «обзор материала»;
— общие подписи «книга», «документ», «untitled».

Если на титуле названия нет — возьми осмысленную часть имени исходного файла (без .pdf/.docx), не выдумывай тему по тексту.
Официальное название чаще всего в первых строках фрагмента."""

_BOOK_TITLE_DESCRIPTIVE_RE = re.compile(
    r"(?iu)"
    r"(описыва|рассматрива|рассмотр|данн(?:ая|ое)\s+(?:книг|работ|стать)|"
    r"автор[sы]?\s+(?:выделя|описы|рассмат)|как\s+автор|в\s+данном\s+"
    r"фрагмент|кратко\s+о\s+|документ\s+о\s+|текст\s+о\s+|"
    r"обзор\s+материал|не\s+указан|содержани(?:е|я)\s+книг|"
    r"пособие\s+по\s+тем|глава\s+\d|раздел\s+\d|введение\s+в\s+проблем)",
)


def _book_title_filename_hint(source_filename: str) -> str:
    stem = Path(source_filename or "").stem.strip()
    return sanitize_output_stem(stem, max_len=90)


def _book_title_is_descriptive_guess(title: str) -> bool:
    """True, если строка похожа на описание содержания, а не на название издания."""
    t = (title or "").strip()
    if len(t) < 2:
        return True
    if len(t) > 95:
        return True
    if _BOOK_TITLE_DESCRIPTIVE_RE.search(t):
        return True
    if re.search(r"[.!?]\s*$", t) and len(t.split()) >= 6:
        return True
    if re.match(r"(?iu)^(как|что\s+такое|обзор|основы|принципы|методы)\s+", t) and len(t.split()) >= 5:
        return True
    return False


def _llm_book_title_chat_once(
    client: Any,
    model: str,
    excerpt: str,
    source_filename: str,
    extra_note: str = "",
) -> str:
    file_hint = _book_title_filename_hint(source_filename)
    focus = excerpt[:LLM_BOOK_TITLE_FOCUS_CHARS]
    payload = extra_note + (
        "Верни JSON {\"title\": \"...\"} — официальное название издания для имени файла.\n"
        "Сначала ищи название в блоке «Начало текста» (титул); не пересказывай содержание.\n\n"
        f"Имя исходного файла: {source_filename}\n"
        f"Подсказка из имени файла (если титула нет): {file_hint or '(пусто)'}\n\n"
        f"Начало текста ({len(excerpt):,} символов, приоритет первых {len(focus):,}):\n\n"
        f"{focus}\n"
    )
    if len(excerpt) > len(focus):
        payload += (
            f"\n---\nПродолжение (реже содержит титул, {len(excerpt) - len(focus):,} симв.):\n\n"
            + excerpt[len(focus) :]
        )
    return _llm_json_chat_completion(
        client,
        model,
        BOOK_TITLE_JSON_SYSTEM,
        payload,
        temperature=0.1,
        max_tokens=LLM_BOOK_TITLE_MAX_COMPLETION_TOKENS,
    )


def _infer_book_title_llm_attempt(
    client: Any,
    model: str,
    excerpt: str,
    source_filename: str,
    extra_note: str,
    events_queue: Optional[queue.Queue],
) -> str:
    rejected = ""
    for attempt in range(2):
        note = extra_note
        if rejected:
            note += (
                f"Предыдущий ответ отклонён («{rejected[:80]}»): это описание содержания, не название.\n"
                "Верни только официальное название с титула или подсказку из имени файла.\n\n"
            )
        raw = _llm_book_title_chat_once(client, model, excerpt, source_filename, note)
        stem = sanitize_output_stem(_parse_llm_book_title_json(raw))
        if len(stem) < 2:
            if events_queue:
                snippet = re.sub(r"\s+", " ", (raw or ""))[:140]
                events_queue.put(
                    (
                        "log",
                        f"  ИИ-имя книги: не распознан JSON (попытка {attempt + 1}/2)"
                        + (f": «{snippet}»" if snippet else "."),
                    )
                )
            continue
        if _book_title_is_descriptive_guess(stem):
            rejected = stem
            if events_queue:
                events_queue.put(
                    ("log", f"  ИИ-имя книги: отклонено как описание («{stem[:60]}»).")
                )
            continue
        return stem
    file_hint = _book_title_filename_hint(source_filename)
    if file_hint and len(file_hint) >= 2 and not _book_title_is_descriptive_guess(file_hint):
        if events_queue:
            events_queue.put(
                ("log", f"  ИИ-имя книги: подставлено из имени файла («{file_hint}»).")
            )
        return file_hint
    return ""


def _infer_book_title_recursive(
    client: Any,
    model: str,
    excerpt: str,
    source_filename: str,
    events_queue: Optional[queue.Queue],
    split_depth: int,
) -> str:
    excerpt = (excerpt or "").strip()
    if not excerpt:
        return ""
    note = ""
    if split_depth:
        note = "Подзадача: текст укорочен из-за лимита контекста; ищи титул в этом фрагменте.\n\n"
    try:
        return _infer_book_title_llm_attempt(
            client, model, excerpt, source_filename, note, events_queue
        )
    except Exception as e:
        if not _is_context_overflow_error(e) or split_depth >= LLM_CONTEXT_SPLIT_MAX_DEPTH:
            if events_queue:
                events_queue.put(("log", f"  ИИ-имя книги: ошибка API: {e}"))
            return ""
        halves = _bisect_text_for_llm_retry(excerpt, min_chars=200)
        if not halves:
            if events_queue:
                events_queue.put(
                    ("log", "  ИИ-имя книги: не удалось разделить текст при переполнении контекста.")
                )
            return ""
        a, b = halves
        if events_queue:
            events_queue.put(
                (
                    "log",
                    f"  ИИ-имя книги: лимит контекста — делю промпт (~{len(excerpt):,} симв.) "
                    f"→ ~{len(a):,} + ~{len(b):,} симв.",
                )
            )
        for part in (a, b):
            got = _infer_book_title_recursive(
                client,
                model,
                part,
                source_filename,
                events_queue,
                split_depth + 1,
            )
            if got and not _book_title_is_descriptive_guess(got):
                return got
        return ""


def _parse_llm_book_title_json(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    candidates = [s]
    stripped = _strip_llm_thinking_blocks(s)
    if stripped and stripped != s:
        candidates.append(stripped)
    for candidate in candidates:
        title = _parse_llm_book_title_json_candidate(candidate)
        if title:
            return title
    return ""


def _parse_llm_book_title_json_candidate(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        lo, hi = s.find("{"), s.rfind("}")
        if lo == -1 or hi <= lo:
            data = None
        else:
            try:
                data = json.loads(s[lo : hi + 1])
            except json.JSONDecodeError:
                data = None
        if data is None:
            m = re.search(
                r'["\']title["\']\s*:\s*["\']((?:\\.|[^"\\\'])*)["\']',
                s,
                flags=re.IGNORECASE,
            )
            if m:
                try:
                    return json.loads(f'"{m.group(1)}"')
                except json.JSONDecodeError:
                    return m.group(1).replace("\\\"", '"').strip()
            if (
                "{" not in s
                and "\n" not in s
                and 2 <= len(s) <= 90
                and not re.match(r'^["\']?title["\']?\s*:', s, re.IGNORECASE)
            ):
                return s
            return ""
    if not isinstance(data, dict):
        return ""
    t = data.get("title")
    if t is None:
        return ""
    return str(t).strip()


def llm_infer_book_title(
    client: Any,
    model: str,
    text: str,
    source_filename: str,
    events_queue: Optional[queue.Queue] = None,
) -> str:
    """Короткое имя для base_stem (без расширения); пустая строка при сбое или пустом тексте."""
    excerpt = (text or "")[:LLM_BOOK_TITLE_MAX_INPUT_CHARS]
    if not excerpt.strip():
        return ""
    if events_queue:
        events_queue.put(
            ("log", f"  ИИ-имя книги: в промпт ~{len(excerpt):,} символов; ожидается JSON с title.")
        )
    stem = _infer_book_title_recursive(client, model, excerpt, source_filename, events_queue, 0)
    if len(stem) < 2:
        if events_queue:
            events_queue.put(("log", "  ИИ-имя: ответ пустой или слишком короткий."))
        return ""
    if events_queue:
        events_queue.put(("log", f"  ИИ-имя: «{stem}»"))
    return stem


def _normalize_inline_ws(fragment: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", (fragment or "").strip())


def _find_toc_anchor(full_text: str, anchor: str, search_from: int) -> int:
    """Индекс начала якоря или -1. Пробует точное совпадение, первую строку, ослабленные пробелы."""
    a = (anchor or "").strip()
    if not a:
        return -1
    window = full_text[search_from:]

    def _locate(needle: str) -> int:
        n = (needle or "").strip()
        if not n:
            return -1
        idx = window.find(n)
        if idx != -1:
            return search_from + idx
        if len(n) > 8 and n.isascii():
            low = window.lower()
            nlow = n.lower()
            idx = low.find(nlow)
            if idx != -1:
                return search_from + idx
        return -1

    hit = _locate(a)
    if hit != -1:
        return hit
    first_line = a.split("\n", 1)[0].strip()
    if first_line and first_line != a:
        hit = _locate(first_line)
        if hit != -1:
            return hit
    if first_line.startswith("#"):
        bare = re.sub(r"^#+\s*", "", first_line).strip()
        if bare:
            hit = _locate(bare)
            if hit != -1:
                return hit
    if len(first_line) > 12:
        loose = _normalize_inline_ws(first_line)
        if loose != first_line:
            hit = _locate(loose)
            if hit != -1:
                return hit
    parts = re.split(r"\s+", first_line[:240].strip())
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        try:
            pat = r"\s+".join(re.escape(p) for p in parts)
            m = re.search(pat, window, flags=re.DOTALL | re.IGNORECASE)
            if m:
                return search_from + m.start()
        except re.error:
            pass
    return -1


def split_by_toc_anchor_strings(
    full_text: str,
    items: list[tuple[str, str]],
    events_queue: Optional[queue.Queue] = None,
) -> list[tuple[str, str]]:
    """items: (anchor, section_title). Возвращает (section_title, chunk)."""
    found: list[tuple[int, str, str]] = []
    search_from = 0
    missed = 0
    for anchor, title in items:
        anchor = (anchor or "").strip()
        title = (title or "").strip()
        if not anchor:
            continue
        idx = _find_toc_anchor(full_text, anchor, search_from)
        if idx == -1 and title:
            idx = _find_toc_anchor(full_text, title, search_from)
        if idx == -1:
            missed += 1
            continue
        found.append((idx, anchor, title or _short_section_title(anchor)))
        search_from = idx + max(len(anchor), 1)

    if events_queue and missed:
        events_queue.put(("log", f"  LLM TOC: якорей не найдено в тексте: {missed} (остальные применены)."))

    if not found:
        return []

    sections: list[tuple[str, str]] = []
    if found[0][0] > 0:
        pre = full_text[: found[0][0]].strip()
        if pre:
            sections.append(("_preface", pre))
    for i, (idx, _anchor, title) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(full_text)
        chunk = full_text[idx:end].strip()
        if chunk:
            sections.append((title, chunk))
    return sections


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = name.strip("._")
    return name[:max_len] or "section"


_NOISE_MD_HEADING = re.compile(
    r"^(page|стр\.?|страница|лист\.?)\s*:?\s*\d+\s*$",
    re.IGNORECASE,
)
_HEURISTIC_TOC_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_HEURISTIC_TOC_CHAPTER_LINE = re.compile(
    r"(?im)^("
    r"(?:Глава|ГЛАВА|Chapter|CHAPTER|Раздел|РАЗДЕЛ|Часть|ЧАСТЬ|Section|SECTION|Приложение|ПРИЛОЖЕНИЕ)"
    r"(?:\s+[\dIVXLC]+)?"
    r"(?:\s*[-—:.]\s*)?"
    r"[^\n]{0,120}"
    r")$"
)
_HEURISTIC_TOC_NUMBERED_LINE = re.compile(
    r"(?im)^(\d{1,2}(?:\.\d{1,2}){1,3}\s+[А-ЯЁA-Z][^\n]{0,120})$"
)
_NOISE_TOC_ANCHOR = re.compile(
    r"(?im)^(?:"
    r"(?:рис(?:унок)?|fig(?:ure)?|таб(?:лица)?|table)\s*[\d.]"
    r"|page\s*\d+|стр\.?\s*\d+|страница\s*\d+"
    r")"
)


def _is_noise_toc_anchor(anchor: str) -> bool:
    line = (anchor or "").split("\n", 1)[0].strip()
    if not line:
        return True
    if _NOISE_TOC_ANCHOR.match(line):
        return True
    bare = line.lstrip("#").strip()
    if _is_noise_markdown_heading(bare):
        return True
    return False


def infer_toc_sections_heuristic(
    text: str,
    events_queue: Optional[queue.Queue] = None,
) -> list[tuple[str, str]]:
    """Якоря разделов по markdown-заголовкам и типичным строкам «Глава N» в OCR-тексте."""
    raw_text = text or ""
    if not raw_text.strip():
        return []

    candidates: list[tuple[int, str, str]] = []
    seen_positions: list[int] = []

    def add_match(pos: int, anchor: str) -> None:
        anchor = (anchor or "").strip()
        if not anchor or _is_noise_toc_anchor(anchor):
            return
        for p in seen_positions:
            if abs(p - pos) < 24:
                return
        seen_positions.append(pos)
        candidates.append((pos, anchor, _short_section_title(anchor)))

    for match in _HEURISTIC_TOC_MD_HEADING.finditer(raw_text):
        title = match.group(2).strip()
        if _is_noise_markdown_heading(title):
            continue
        add_match(match.start(), match.group(0).strip())

    for pattern in (_HEURISTIC_TOC_CHAPTER_LINE, _HEURISTIC_TOC_NUMBERED_LINE):
        for match in pattern.finditer(raw_text):
            add_match(match.start(), match.group(1).strip())

    candidates.sort(key=lambda x: x[0])
    out = [(anchor, title) for _, anchor, title in candidates]
    if events_queue and out:
        events_queue.put(("log", f"  TOC (эвристика): найдено якорей в тексте: {len(out)}."))
    return out


def _split_sections_with_fallbacks(
    text: str,
    preferred_level: int,
    events_queue: Optional[queue.Queue] = None,
) -> list[tuple[str, str]]:
    """Разбиение по markdown-заголовкам: предпочтительный уровень, затем H1–H3."""
    levels: list[int] = []
    for lvl in (preferred_level, 1, 2, 3):
        if lvl not in levels:
            levels.append(lvl)
    for level in levels:
        sections = split_markdown_by_heading(text, level=level)
        if len(sections) >= 2:
            if events_queue and level != preferred_level:
                events_queue.put(
                    ("log", f"  разбиение по заголовкам H{level} (запасной уровень).")
                )
            return sections
    return split_markdown_by_heading(text, level=preferred_level)


def _is_noise_markdown_heading(title: str) -> bool:
    """Служебные заголовки OCR/markdown (номер страницы), не настоящие главы."""
    t = (title or "").strip()
    if not t:
        return True
    if _NOISE_MD_HEADING.match(t):
        return True
    if re.fullmatch(r"\d{1,4}", t):
        return True
    return False


def _content_leading_line_starts_with_heading(content: str) -> bool:
    for line in (content or "").splitlines():
        s = line.strip()
        if not s:
            continue
        return s.startswith("#")
    return False


def split_markdown_by_heading(
    text: str,
    level: int = 2,
    skip_noise_headings: bool = True,
) -> list[tuple[str, str]]:
    pattern = re.compile(r"^(#{1," + str(level) + r"})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    last_pos = 0
    last_title = "_preface"

    for match in pattern.finditer(text):
        hashes = match.group(1)
        title = match.group(2).strip()
        if len(hashes) != level:
            continue
        if skip_noise_headings and _is_noise_markdown_heading(title):
            continue
        chunk = text[last_pos:match.start()].strip()
        if chunk or last_title != "_preface":
            sections.append((last_title, chunk))
        last_pos = match.start()
        last_title = title

    chunk = text[last_pos:].strip()
    if chunk:
        sections.append((last_title, chunk))

    return sections


def save_split_sections(
    sections: list[tuple[str, str]],
    out_dir: Path,
    base_stem: str,
    events_queue: queue.Queue,
    inject_md_h2_from_title: bool = False,
    obsidian_links: bool = False,
) -> int:
    split_dir = out_dir / base_stem
    split_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    planned: list[tuple[str, Path, str, str]] = []

    for idx, (title, content) in enumerate(sections, start=1):
        safe_title = _sanitize_filename(title)
        filename = f"{idx:03d}_{safe_title}.md"
        file_path = split_dir / filename

        counter = 1
        while file_path.exists():
            file_path = split_dir / f"{idx:03d}_{safe_title}_{counter}.md"
            counter += 1
        planned.append((title, file_path, file_path.name, content))

    if obsidian_links:
        index_lines = [f"# {base_stem}", "", "## Разделы", ""]
        for title, _file_path, filename, _content in planned:
            index_lines.append(f"- [[{Path(filename).stem}|{title}]]")
        (split_dir / "000_index.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")
        events_queue.put(("log", f"    obsidian index → {(split_dir / '000_index.md').relative_to(out_dir)}"))

    for pos, (title, file_path, filename, content) in enumerate(planned):
        body = content
        if (
            inject_md_h2_from_title
            and title != "_preface"
            and (content or "").strip()
            and not _content_leading_line_starts_with_heading(content)
        ):
            h = (title or "").split("\n", 1)[0].strip()
            if h:
                body = f"## {h}\n\n{content}"
        if obsidian_links:
            prev_file = Path(planned[pos - 1][2]).stem if pos > 0 else ""
            next_file = Path(planned[pos + 1][2]).stem if pos + 1 < len(planned) else ""
            nav_parts = ["[[000_index|Оглавление]]"]
            if prev_file:
                nav_parts.append(f"[[{prev_file}|← предыдущий раздел]]")
            if next_file:
                nav_parts.append(f"[[{next_file}|следующий раздел →]]")
            frontmatter = (
                "---\n"
                f"llmmd_book: \"{base_stem}\"\n"
                f"llmmd_section: \"{title}\"\n"
                "---\n\n"
            )
            body = frontmatter + " · ".join(nav_parts) + "\n\n" + body

        file_path.write_text(body, encoding="utf-8")
        events_queue.put(("log", f"    сплит → {file_path.relative_to(out_dir)}"))
        saved += 1

    return saved


def write_book_report(
    out_dir: Path,
    artifact_stem: str,
    *,
    source: Path,
    title_stem: str,
    stem_how: str,
    outputs: list[Path],
    stages: list[dict[str, Any]],
    figures_count: int,
    sections_count: int | None,
    stopped: bool,
) -> Path:
    report_path = out_dir / f"{artifact_stem}_report.md"
    lines: list[str] = [
        f"# Отчёт обработки: {artifact_stem}",
        "",
        f"- Исходный файл: `{source}`",
        f"- Имя из документа: `{title_stem}` ({stem_how or 'не указано'})",
        f"- Итоговый stem артефактов: `{artifact_stem}`",
        f"- Состояние: {'остановлено пользователем, сохранён частичный результат' if stopped else 'завершено'}",
        f"- Извлечено изображений: {figures_count}",
        f"- Секций разбиения: {sections_count if sections_count is not None else 'не выполнялось'}",
        "",
        "## Файлы",
        "",
    ]
    if outputs:
        for path in outputs:
            try:
                shown = path.relative_to(out_dir)
            except ValueError:
                shown = path
            lines.append(f"- `{shown}`")
    else:
        lines.append("- Файлы результата не записаны.")
    lines.extend(["", "## Этапы", ""])
    if stages:
        for item in stages:
            duration = item.get("duration_s")
            duration_text = format_duration_ru(duration) if isinstance(duration, (int, float)) else "—"
            status = item.get("status") or ""
            label = item.get("label") or item.get("stage") or "этап"
            details = item.get("details") or ""
            suffix = f" — {details}" if details else ""
            lines.append(f"- {label}: {status}, {duration_text}{suffix}")
    else:
        lines.append("- Нет данных по этапам.")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


LLM_HTTP_TIMEOUT_S = 600.0
LLM_HTTP_CONNECT_TIMEOUT_S = 30.0


def _make_openai_client(base_url: str, api_key: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("Пакет openai не установлен. Установите: pip install openai")
    kwargs: dict = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "dummy-key")}
    if httpx is not None:
        kwargs["timeout"] = httpx.Timeout(LLM_HTTP_TIMEOUT_S, connect=LLM_HTTP_CONNECT_TIMEOUT_S)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_markitdown(
    use_plugins: bool,
    use_llm: bool,
    model_name: str,
    base_url: str,
    api_key: str,
    ocr_model_name: str = "",
    ocr_base_url: str = "",
    ocr_api_key: str = "",
) -> tuple[Any, Any, str, str]:
    """
    Возвращает (md_instance, semantic_client, semantic_model, ocr_mode).

    semantic_client/model — для LaTeX, TOC, описаний рисунков, названия книги.
    Если задан ocr_model_name — MarkItDown получает отдельный OCR-клиент (быстрая/дешёвая модель);
    иначе MarkItDown тоже использует semantic_client.
    """
    if MarkItDown is None:
        raise RuntimeError("Пакет markitdown не установлен. Установите зависимости проекта.")
    kwargs: dict = {"enable_plugins": use_plugins}
    semantic_client = None
    semantic_model = model_name or "gpt-4o"
    ocr_mode = "skipped"

    if use_llm:
        semantic_client = _make_openai_client(base_url, api_key)

        use_separate_ocr = bool(ocr_model_name and ocr_model_name.strip())
        if use_separate_ocr:
            ocr_client = _make_openai_client(
                ocr_base_url or base_url,
                ocr_api_key or api_key,
            )
            kwargs["llm_client"] = ocr_client
            kwargs["llm_model"] = ocr_model_name.strip()
            ocr_mode = f"OCR={ocr_model_name.strip()} · semantic={semantic_model}"
        else:
            kwargs["llm_client"] = semantic_client
            kwargs["llm_model"] = semantic_model
            ocr_mode = f"enabled ({semantic_model})"

    return MarkItDown(**kwargs), semantic_client, semantic_model, ocr_mode


def run_conversion_job(args: dict, events: queue.Queue) -> None:
    try:
        llm_text, ocr_explicit, _, llm_figure = resolve_llm_models(args)
        stage_started: dict[str, float] = {}
        stage_reports: dict[int, list[dict[str, Any]]] = {}
        enabled_stages = normalize_enabled_stages(args.get("enabled_stages"))
        pause_event: Optional[threading.Event] = args.get("pause_event")

        def _stage_enabled(stage: str) -> bool:
            return stage in enabled_stages

        def _wait_if_paused(label: str) -> None:
            if pause_event is None or not pause_event.is_set():
                return
            events.put(("status", f"Пауза: {label}"))
            events.put(("log", f"  пауза перед этапом «{label}»."))
            while pause_event.is_set() and not args["cancel_event"].is_set():
                pause_event.wait(0.25)
            if args["cancel_event"].is_set():
                return
            events.put(("log", "  пауза снята, продолжаю."))

        def _stage_id(stage: str, file_index: int | None = None) -> str:
            prefix = str(file_index) if file_index is not None else "job"
            return f"{prefix}:{stage}"

        def _stage_start(
            stage: str,
            label: str,
            *,
            file_path: Path | None = None,
            file_index: int | None = None,
            files_total: int | None = None,
            model: str | None = None,
            details: str = "",
        ) -> str:
            sid = _stage_id(stage, file_index)
            started = time.time()
            stage_started[sid] = started
            _stage_event(
                events,
                stage_id=sid,
                status="running",
                stage=stage,
                label=label,
                file_path=str(file_path) if file_path else None,
                file_index=file_index,
                files_total=files_total,
                model=model,
                details=details,
                started_at_unix=started,
            )
            events.put(("status", label))
            return sid

        def _stage_finish(
            stage: str,
            label: str,
            *,
            file_path: Path | None = None,
            file_index: int | None = None,
            files_total: int | None = None,
            model: str | None = None,
            details: str = "",
            status: str = "done",
        ) -> None:
            sid = _stage_id(stage, file_index)
            started = stage_started.get(sid)
            duration = round(time.time() - started, 3) if started else None
            _stage_event(
                events,
                stage_id=sid,
                status=status,
                stage=stage,
                label=label,
                file_path=str(file_path) if file_path else None,
                file_index=file_index,
                files_total=files_total,
                model=model,
                details=details,
                started_at_unix=started,
                duration_s=duration,
            )
            if file_index is not None:
                stage_reports.setdefault(file_index, []).append(
                    {
                        "stage": stage,
                        "label": label,
                        "status": status,
                        "details": details,
                        "duration_s": duration,
                    }
                )
            if status == "done" and duration is not None:
                events.put(("log", f"  этап «{label}»: {format_duration_ru(duration)}"))

        def _stage_skip(
            stage: str,
            label: str,
            *,
            file_path: Path | None = None,
            file_index: int | None = None,
            files_total: int | None = None,
            model: str | None = None,
            details: str = "",
        ) -> None:
            _stage_event(
                events,
                stage_id=_stage_id(stage, file_index),
                status="skipped",
                stage=stage,
                label=label,
                file_path=str(file_path) if file_path else None,
                file_index=file_index,
                files_total=files_total,
                model=model,
                details=details,
            )
            if file_index is not None:
                stage_reports.setdefault(file_index, []).append(
                    {
                        "stage": stage,
                        "label": label,
                        "status": "skipped",
                        "details": details,
                        "duration_s": None,
                    }
                )

        def _alog(msg: str) -> None:
            events.put(("log", msg))

        _stage_start("init", "Инициализация OCR/LLM", model=ocr_explicit or llm_text or "")
        if args.get("use_llm") and args.get("use_lmstudio_autoload"):
            try:
                ensure_lmstudio_roles(args, ["ocr_model"], _alog)
            except Exception as e:
                events.put(("log", f"LM Studio autoload (OCR): {e}"))
                _stage_finish(
                    "init",
                    "Инициализация OCR/LLM",
                    model=ocr_explicit or llm_text or "",
                    details=str(e),
                    status="failed",
                )
                events.put(("done", "Ошибка autoload"))
                return

        md, semantic_client, semantic_model, ocr_mode = build_markitdown(
            args["use_plugins"],
            args["use_llm"],
            llm_text,
            args["base_url"],
            args["api_key"],
            ocr_model_name=ocr_explicit,
            ocr_base_url=args.get("ocr_base_url", ""),
            ocr_api_key=args.get("ocr_api_key", ""),
        )
        events.put(("log", f"Инициализация MarkItDown завершена. OCR режим: {ocr_mode}"))
        _stage_finish("init", "Инициализация OCR/LLM", model=ocr_explicit or llm_text or "", details=f"OCR режим: {ocr_mode}")
        set_pdf_cancel_event(args["cancel_event"])
        set_pdf_pause_event(pause_event)
        if args["use_plugins"]:
            install_pdf_cancel_hooks()
        if args["use_plugins"] and args["use_llm"]:
            install_ocr_image_limits()
            install_ocr_gui_logging(events)
            events.put(("log", "Подробный лог OCR включён."))

        out_dir = Path(args["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        files = list(args["files"])
        if not files:
            events.put(("log", "Нет файлов для обработки."))
            events.put(("done", "Нечего обрабатывать"))
            return

        events.put(("progress_max", len(files)))
        events.put(("log", f"Файлов в очереди: {len(files)}"))
        if len(files) > 1:
            events.put(
                (
                    "log",
                    "Несколько файлов: каждый обрабатывается отдельно "
                    "(свой markdown / своя папка при разбиении), без объединения в один текст.",
                )
            )

        manual_in = (args.get("output_stem") or "").strip()
        manual_stem = sanitize_output_stem(manual_in)
        if manual_in and not manual_stem:
            events.put(("log", "Название для сохранения после очистки пустое — будет автоимя."))
        if manual_stem and len(files) > 1:
            events.put(
                (
                    "log",
                    "Название для сохранения: в очереди несколько файлов — поле не применяется "
                    "(у каждого файла своё имя).",
                )
            )
            manual_stem = ""
        prefer_meta = bool(args.get("prefer_pdf_metadata_title", True))

        args["cancel_event"].clear()
        cancelled_queue = False
        for index, file_path in enumerate(files, start=1):
            if args["cancel_event"].is_set():
                cancelled_queue = True
                events.put(("log", "Очередь остановлена пользователем; следующие файлы не запускаются."))
                break
            src = Path(file_path)
            report_outputs: list[Path] = []
            figures_count = 0
            sections_count: int | None = None
            file_stopped = False
            events.put(("current", str(src)))
            events.put(("ocr", f"{'включён' if args['use_llm'] else 'выключен / будет пропущен'}"))
            events.put(("log", f"[{index}/{len(files)}] Обработка: {src}"))
            try:
                _stage_start(
                    "inspect",
                    "Подготовка файла",
                    file_path=src,
                    file_index=index,
                    files_total=len(files),
                )
                try:
                    sz = src.stat().st_size
                    events.put(("log", f"  размер файла: {sz:,} байт ({sz / (1024*1024):.2f} МиБ)"))
                except OSError:
                    sz = None

                pages_tuple = None
                stem_extra = ""
                pdf_spec = (args.get("pdf_pages_spec") or "").strip()
                if src.suffix.lower() == ".pdf" and pdf_spec:
                    try:
                        pages_tuple = parse_pdf_page_spec(pdf_spec)
                    except ValueError as e:
                        events.put(("log", f"  пропуск из-за страниц PDF: {e}"))
                        raise
                    stem_extra = pdf_pages_filename_suffix(pages_tuple)
                    events.put(("log", f"  PDF: обрабатываются только страницы {pages_tuple}"))
                _stage_finish(
                    "inspect",
                    "Подготовка файла",
                    file_path=src,
                    file_index=index,
                    files_total=len(files),
                    details=f"{sz:,} байт" if isinstance(sz, int) else "",
                )

                raster_profile: Optional[PdfRasterProfile] = None
                if src.suffix.lower() == ".pdf":
                    try:
                        page_set_probe = set(pages_tuple) if pages_tuple else None
                        raster_profile = probe_pdf_raster_profile(src, page_set_probe)
                        events.put(
                            (
                                "log",
                                f"  PDF растровый профиль: {raster_profile.kind} — {raster_profile.message}",
                            )
                        )
                    except Exception as e:
                        events.put(("log", f"  PDF растровый профиль: не удалось определить ({e})"))

                _wait_if_paused("Конвертация/OCR в Markdown")
                if args["cancel_event"].is_set():
                    cancelled_queue = True
                    events.put(("log", "  остановлено перед конвертацией текущего файла."))
                    break
                _stage_start(
                    "convert",
                    "Конвертация/OCR в Markdown",
                    file_path=src,
                    file_index=index,
                    files_total=len(files),
                    model=ocr_explicit or llm_text if args.get("use_llm") else "",
                    details=f"страницы: {pages_tuple}" if pages_tuple else "весь документ",
                )
                set_active_pdf_raster_profile(
                    raster_profile if src.suffix.lower() == ".pdf" else None
                )
                try:
                    if pages_tuple:
                        events.put(("log", "  запуск convert_stream (подмножество страниц)…"))
                        pdf_subset = build_pdf_subset_bytes(src, pages_tuple)
                        stream = io.BytesIO(pdf_subset)
                        si = StreamInfo(extension=".pdf", mimetype="application/pdf", filename=src.name)
                        result = md.convert_stream(stream, stream_info=si)
                    else:
                        events.put(("log", "  запуск convert_local…"))
                        result = md.convert_local(str(src))
                finally:
                    set_active_pdf_raster_profile(None)
                _stage_finish(
                    "convert",
                    "Конвертация/OCR в Markdown",
                    file_path=src,
                    file_index=index,
                    files_total=len(files),
                    model=ocr_explicit or llm_text if args.get("use_llm") else "",
                )

                _stage_start("clean", "Очистка Markdown", file_path=src, file_index=index, files_total=len(files))
                text_raw = result.text_content or ""
                text, cid_removed = strip_pdf_cid_garbage(text_raw)
                if cid_removed:
                    events.put(
                        ("log", f"  удалено строк с мусором pdfminer «(cid:…)»: {cid_removed}")
                    )
                _stage_finish(
                    "clean",
                    "Очистка Markdown",
                    file_path=src,
                    file_index=index,
                    files_total=len(files),
                    details=f"{len(text):,} символов; удалено CID-строк: {cid_removed}",
                )

                stopped = args["cancel_event"].is_set()
                file_stopped = stopped
                if stopped:
                    events.put(("log", "  обработка была остановлена; в markdown сохранено то, что успели получить."))

                need_text_autoload = (
                    args.get("use_llm")
                    and args.get("use_lmstudio_autoload")
                    and semantic_client is not None
                    and (
                        (args.get("formulas_llm_latex") and _stage_enabled("formulas"))
                        or (args.get("split_llm_toc") and _stage_enabled("split"))
                        or (not manual_stem and _stage_enabled("title"))
                    )
                )
                if not stopped and need_text_autoload:
                    _wait_if_paused("Автозагрузка текстовой модели")
                    _stage_start(
                        "text_autoload",
                        "Автозагрузка текстовой модели",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model,
                    )
                    try:
                        ensure_lmstudio_roles(args, ["text_model"], _alog)
                        _stage_finish(
                            "text_autoload",
                            "Автозагрузка текстовой модели",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            model=semantic_model,
                        )
                    except Exception as e:
                        events.put(("log", f"  LM Studio autoload (текст): {e}"))
                        _stage_finish(
                            "text_autoload",
                            "Автозагрузка текстовой модели",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            model=semantic_model,
                            details=str(e),
                            status="failed",
                        )
                        events.put(("done", "Ошибка autoload"))
                        return

                if (
                    not stopped
                    and _stage_enabled("formulas")
                    and args.get("formulas_llm_latex")
                    and args.get("use_llm")
                    and semantic_client is not None
                    and (text or "").strip()
                ):
                    _wait_if_paused("Формулы -> LaTeX")
                    _stage_start(
                        "formulas",
                        "Формулы -> LaTeX",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model,
                    )
                    try:
                        events.put(("log", "  формулы → LaTeX (LLM): преобразование выражений в тексте…"))
                        text = llm_convert_formulas_in_markdown(
                            semantic_client,
                            semantic_model,
                            text,
                            events,
                            args.get("cancel_event"),
                        )
                        _stage_finish(
                            "formulas",
                            "Формулы -> LaTeX",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            model=semantic_model,
                            details=f"{len(text):,} символов после обработки",
                        )
                    except Exception as e:
                        events.put(("log", f"  формулы → LaTeX (LLM): сбой — {e}"))
                        _stage_finish(
                            "formulas",
                            "Формулы -> LaTeX",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            model=semantic_model,
                            details=str(e),
                            status="failed",
                        )
                else:
                    _stage_skip(
                        "formulas",
                        "Формулы -> LaTeX",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model if args.get("use_llm") else "",
                        details="выключено или нет текста",
                    )

                multi_book = len(files) > 1
                base_stem = ""
                stem_how = ""
                if (
                    not stopped
                    and _stage_enabled("title")
                    and args["use_llm"]
                    and semantic_client is not None
                    and not manual_stem
                ):
                    _wait_if_paused("Определение имени книги")
                    _stage_start(
                        "title",
                        "Определение имени книги",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model,
                    )
                    try:
                        hint = "очередь" if multi_book else "одиночный файл"
                        events.put(("log", f"  имя для этой книги: запрос к ИИ ({hint})…"))
                        ai_stem = llm_infer_book_title(
                            semantic_client, semantic_model, text, src.name, events
                        )
                        if ai_stem:
                            base_stem = ai_stem + stem_extra
                            stem_how = "ИИ"
                        _stage_finish(
                            "title",
                            "Определение имени книги",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            model=semantic_model,
                            details=base_stem or "ИИ не вернул имя",
                        )
                    except Exception as e:
                        events.put(("log", f"  ИИ-имя: ошибка ({e}); подставляется автоимя."))
                        _stage_finish(
                            "title",
                            "Определение имени книги",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            model=semantic_model,
                            details=str(e),
                            status="failed",
                        )
                else:
                    _stage_skip(
                        "title",
                        "Определение имени книги",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model if args.get("use_llm") else "",
                        details="остановлено, ручное имя или LLM выключен",
                    )
                if not base_stem:
                    base_stem, stem_how = resolve_output_base_stem(
                        src, stem_extra, manual_stem, prefer_meta
                    )
                events.put(("log", f"  имя для сохранения ({stem_how}): {base_stem}"))
                title_stem = base_stem
                artifact_stem = unique_output_stem(out_dir, base_stem)
                if artifact_stem != base_stem:
                    events.put(
                        (
                            "log",
                            f"  найден конфликт имён — артефакты этой книги будут сохранены как: {artifact_stem}",
                        )
                    )
                base_stem = artifact_stem

                figures_appendix = ""
                if not stopped and _stage_enabled("images_extract") and src.suffix.lower() == ".pdf" and args.get("extract_pdf_images"):
                    assets_name = f"{base_stem}_assets"
                    page_set = set(pages_tuple) if pages_tuple else None
                    skip_figures = raster_profile is not None and raster_profile.skip_figure_extract
                    try:
                        if skip_figures:
                            events.put(
                                (
                                    "log",
                                    "  извлечение рисунков пропущено (быстрее для скан/лавины xobject): "
                                    + (raster_profile.message if raster_profile else ""),
                                )
                            )
                            _stage_skip(
                                "images_extract",
                                "Извлечение изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                                details=raster_profile.message if raster_profile else "skip",
                            )
                            _stage_skip(
                                "images_describe",
                                "Описание изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                                details="извлечение пропущено",
                            )
                            figs = []
                        else:
                            _wait_if_paused("Извлечение изображений")
                            _stage_start(
                                "images_extract",
                                "Извлечение изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                            )
                            assets_dir = out_dir / assets_name
                            events.put(("log", f"  папка иллюстраций: {assets_dir}"))
                            figs = extract_figures_from_pdf(src, out_dir, assets_name, page_set)
                            figures_count = len(figs)
                            if figures_count:
                                report_outputs.append(out_dir / assets_name)
                            events.put(("log", f"  PDF: извлечено изображений (крупных): {len(figs)}"))
                            _stage_finish(
                                "images_extract",
                                "Извлечение изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                                details=f"{len(figs)} изображений",
                            )
                        desc_map: Optional[dict[tuple[int, int], str]] = None
                        if (
                            not skip_figures
                            and figs
                            and _stage_enabled("images_describe")
                            and args.get("describe_figures_llm")
                            and args.get("use_llm")
                            and semantic_client is not None
                        ):
                            if args.get("use_lmstudio_autoload"):
                                _wait_if_paused("Автозагрузка модели рисунков")
                                _stage_start(
                                    "figure_autoload",
                                    "Автозагрузка модели рисунков",
                                    file_path=src,
                                    file_index=index,
                                    files_total=len(files),
                                    model=llm_figure,
                                )
                                try:
                                    ensure_lmstudio_roles(args, ["figure_model"], _alog)
                                    _stage_finish(
                                        "figure_autoload",
                                        "Автозагрузка модели рисунков",
                                        file_path=src,
                                        file_index=index,
                                        files_total=len(files),
                                        model=llm_figure,
                                    )
                                except Exception as e:
                                    events.put(("log", f"  LM Studio autoload (рисунки): {e}"))
                                    _stage_finish(
                                        "figure_autoload",
                                        "Автозагрузка модели рисунков",
                                        file_path=src,
                                        file_index=index,
                                        files_total=len(files),
                                        model=llm_figure,
                                        details=str(e),
                                        status="failed",
                                    )
                                    raise
                            _wait_if_paused("Описание изображений")
                            _stage_start(
                                "images_describe",
                                "Описание изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                                model=llm_figure,
                                details=f"{len(figs)} изображений",
                            )
                            desc_map = collect_figure_descriptions(
                                semantic_client,
                                llm_figure,
                                figs,
                                out_dir,
                                args.get("cancel_event"),
                                lambda m: events.put(("log", m)),
                                max_workers=args.get("figures_workers", 4),
                            )
                            _stage_finish(
                                "images_describe",
                                "Описание изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                                model=llm_figure,
                                details=f"{len(desc_map or {})}/{len(figs)} описаний",
                            )
                        elif not skip_figures:
                            _stage_skip(
                                "images_describe",
                                "Описание изображений",
                                file_path=src,
                                file_index=index,
                                files_total=len(files),
                                model=llm_figure if args.get("use_llm") else "",
                                details="нет изображений или описание выключено",
                            )
                        if figs and not skip_figures:
                            figures_appendix = build_figures_markdown_section(
                                base_stem, figs, desc_map
                            )
                    except Exception as e:
                        events.put(("log", f"  извлечение иллюстраций PDF: {e}"))
                        _stage_event(
                            events,
                            stage_id=_stage_id("images_pipeline_error", index),
                            status="failed",
                            stage="images_pipeline_error",
                            label="Извлечение/описание изображений",
                            file_path=src,
                            file_index=index,
                            files_total=len(files),
                            details=str(e),
                        )
                else:
                    _stage_skip(
                        "images_extract",
                        "Извлечение изображений",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        details="остановлено, не PDF или выключено",
                    )
                    _stage_skip(
                        "images_describe",
                        "Описание изображений",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        details="нет этапа извлечения",
                    )

                combined_writes = _stage_enabled("write_combined") and ((not args["do_split"]) or args["keep_combined"])
                if combined_writes:
                    _stage_start("write_combined", "Сохранение общего Markdown", file_path=src, file_index=index, files_total=len(files))
                    out_name = base_stem + ".md"
                    out_path = out_dir / out_name
                    suffix = 1
                    while out_path.exists():
                        out_path = out_dir / f"{base_stem}_{suffix}.md"
                        suffix += 1
                    body = text
                    if figures_appendix:
                        meta = (
                            "<!-- llmmd-meta: в конце файла — раздел «Иллюстрации и рисунки» "
                            "с превью, путями к файлам и подсказками для ИИ. -->\n\n"
                        )
                        body = meta + text + "\n\n" + figures_appendix
                    out_path.write_text(body, encoding="utf-8")
                    report_outputs.append(out_path)
                    events.put(("log", f"  общий файл → {out_path.name} ({len(body):,} символов)"))
                    _stage_finish(
                        "write_combined",
                        "Сохранение общего Markdown",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        details=f"{out_path.name}; {len(body):,} символов",
                    )
                else:
                    _stage_skip(
                        "write_combined",
                        "Сохранение общего Markdown",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        details="выключено: сохраняются только секции",
                    )

                if not stopped and _stage_enabled("split") and args["do_split"] and text.strip():
                    _wait_if_paused("Разбиение Markdown на секции")
                    _stage_start(
                        "split",
                        "Разбиение Markdown на секции",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model if args.get("split_llm_toc") and args.get("use_llm") else "",
                    )
                    sections: list[tuple[str, str]] = []
                    split_used_llm_toc = False
                    toc_items: list[tuple[str, str]] = []
                    try:
                        if args.get("split_llm_toc") and args.get("use_llm") and semantic_client is not None:
                            events.put(
                                (
                                    "log",
                                    "  разбиение: эвристика по тексту + LLM (якоря и заголовки)…",
                                )
                            )
                            toc_items = infer_toc_sections_combined(
                                semantic_client,
                                semantic_model,
                                text,
                                events,
                                args.get("cancel_event"),
                                use_llm=True,
                            )
                        else:
                            events.put(("log", "  разбиение: эвристика по структуре текста (без LLM)…"))
                            toc_items = infer_toc_sections_heuristic(text, events)

                        if toc_items:
                            sections = split_by_toc_anchor_strings(text, toc_items, events)
                            events.put(
                                (
                                    "log",
                                    f"  якорей для нарезки: {len(toc_items)}; "
                                    f"секций после сопоставления с текстом: {len(sections)}",
                                )
                            )
                            if len(sections) >= 2:
                                split_used_llm_toc = True
                            elif sections:
                                events.put(
                                    (
                                        "log",
                                        "  найдена только одна секция по якорям — пробуем заголовки markdown.",
                                    )
                                )
                                sections = []
                    except Exception as e:
                        events.put(("log", f"  оглавление/якоря: {e}; откат на заголовки markdown."))
                        sections = []
                        toc_items = []

                    if not sections:
                        level = args["split_level"]
                        events.put(("log", f"  разбиение по заголовкам H{level}…"))
                        sections = _split_sections_with_fallbacks(text, level, events)

                    n_sections = len(sections)
                    sections_count = n_sections
                    events.put(("log", f"  найдено секций: {n_sections}"))
                    if n_sections > 0:
                        saved = save_split_sections(
                            sections,
                            out_dir,
                            base_stem,
                            events,
                            inject_md_h2_from_title=split_used_llm_toc,
                            obsidian_links=bool(args.get("obsidian_links")),
                        )
                        split_dir = out_dir / base_stem
                        report_outputs.append(split_dir)
                        if args.get("obsidian_links"):
                            report_outputs.append(split_dir / "000_index.md")
                        events.put(("log", f"  сохранено подфайлов: {saved} → {split_dir}/"))
                        if figures_appendix.strip():
                            split_dir.mkdir(parents=True, exist_ok=True)
                            fig_index = split_dir / "000_figures.md"
                            fig_index.write_text(figures_appendix, encoding="utf-8")
                            report_outputs.append(fig_index)
                            events.put(("log", f"    иллюстрации (индекс) → {fig_index.relative_to(out_dir)}"))
                    else:
                        events.put(("log", "  секций не найдено — возможно, нет заголовков нужного уровня."))
                        if figures_appendix.strip():
                            fig_only = out_dir / f"{base_stem}_figures.md"
                            fig_only.write_text(figures_appendix, encoding="utf-8")
                            report_outputs.append(fig_only)
                            events.put(("log", f"  иллюстрации (отдельный файл) → {fig_only.name}"))
                    _stage_finish(
                        "split",
                        "Разбиение Markdown на секции",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        model=semantic_model if args.get("split_llm_toc") and args.get("use_llm") else "",
                        details=f"{n_sections} секций",
                    )
                else:
                    _stage_skip(
                        "split",
                        "Разбиение Markdown на секции",
                        file_path=src,
                        file_index=index,
                        files_total=len(files),
                        details="остановлено, выключено или пустой текст",
                    )

                _stage_start("preview", "Подготовка превью", file_path=src, file_index=index, files_total=len(files))
                preview_src = text if not figures_appendix else text + "\n\n" + figures_appendix
                cap = 19_500 if multi_book else 20_000
                if multi_book:
                    preview_hdr = (
                        f"**Книга {index} из {len(files)}** — `{src.name}` → выход: `{base_stem}`\n\n---\n\n"
                    )
                    rest = max(0, cap - len(preview_hdr))
                    events.put(("preview", preview_hdr + preview_src[:rest]))
                else:
                    events.put(("preview", preview_src[:cap]))
                _stage_finish(
                    "preview",
                    "Подготовка превью",
                    file_path=src,
                    file_index=index,
                    files_total=len(files),
                    details=f"выход: {base_stem}",
                )
                report_path = write_book_report(
                    out_dir,
                    base_stem,
                    source=src,
                    title_stem=title_stem,
                    stem_how=stem_how,
                    outputs=report_outputs,
                    stages=stage_reports.get(index, []),
                    figures_count=figures_count,
                    sections_count=sections_count,
                    stopped=file_stopped,
                )
                events.put(("log", f"  отчёт по книге → {report_path.name}"))

            except Exception as e:
                tb = traceback.format_exc(limit=3)
                events.put(("log", f"Ошибка для {src.name}: {e}"))
                events.put(("log", tb))
                _stage_event(
                    events,
                    stage_id=_stage_id("file_error", index),
                    status="failed",
                    stage="file_error",
                    label="Обработка файла",
                    file_path=str(src),
                    file_index=index,
                    files_total=len(files),
                    details=str(e),
                )
            finally:
                events.put(("progress", index))

        if cancelled_queue or args["cancel_event"].is_set():
            events.put(("done", "Остановлено пользователем"))
        else:
            events.put(("done", "Обработка завершена"))
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        events.put(("log", f"Критическая ошибка: {e}"))
        events.put(("log", tb))
        events.put(("done", "Остановлено с ошибкой"))
    finally:
        set_pdf_cancel_event(None)
        set_pdf_pause_event(None)
        events.put(("stop_btn", "disabled"))
