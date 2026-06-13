"""Извлечение встроенных изображений из PDF и блок Markdown для читателя и ИИ."""

from __future__ import annotations

import base64
import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from gui.services.pdf_raster_profile import (
    FRAGMENT_FLOOD_PER_PAGE,
    MAX_FIGURES_EXTRACT_TOTAL,
    probe_pdf_raster_profile,
)

# Отсекаем мелкие декоративные/иконки (точки, линии).
_MIN_WIDTH = 48
_MIN_HEIGHT = 48
_MIN_PIXELS = 4000


@dataclass
class ExtractedFigure:
    page: int
    index_on_page: int
    filename: str
    width: int
    height: int
    rel_posix: str  # от .md файла в out_dir, например book_assets/p3_02.png


def _posix_rel(parent: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(parent.resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def extract_figures_from_pdf(
    pdf_path: Path,
    out_dir: Path,
    assets_folder_name: str,
    pages_filter: Optional[set[int]] = None,
) -> list[ExtractedFigure]:
    """
    Сохраняет растровые изображения со страниц PDF в out_dir / assets_folder_name /.
    pages_filter: множество номеров страниц (1-based) или None = все страницы.
    """
    import fitz

    quick = probe_pdf_raster_profile(pdf_path, pages_filter)
    if quick.skip_figure_extract:
        return []

    assets_dir = out_dir / assets_folder_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    figures: list[ExtractedFigure] = []
    try:
        for page_idx in range(doc.page_count):
            if len(figures) >= MAX_FIGURES_EXTRACT_TOTAL:
                break
            page_num = page_idx + 1
            if pages_filter is not None and page_num not in pages_filter:
                continue
            page = doc[page_idx]
            seen_xrefs: set[int] = set()
            img_list = page.get_images(full=True) or []
            if len({int(i[0]) for i in img_list}) > FRAGMENT_FLOOD_PER_PAGE:
                continue
            local_i = 0
            for info in img_list:
                if len(figures) >= MAX_FIGURES_EXTRACT_TOTAL:
                    break
                xref = int(info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    img_dict = doc.extract_image(xref)
                except Exception:
                    continue
                w = int(img_dict.get("width") or 0)
                h = int(img_dict.get("height") or 0)
                if w < _MIN_WIDTH or h < _MIN_HEIGHT or w * h < _MIN_PIXELS:
                    continue
                ext = (img_dict.get("ext") or "png").lower()
                if ext not in ("png", "jpeg", "jpg", "webp", "bmp", "gif"):
                    ext = "png"
                local_i += 1
                fname = f"p{page_num}_{local_i:02d}.{ext}"
                dest = assets_dir / fname
                image_bytes = img_dict.get("image")
                if not image_bytes:
                    continue
                dest.write_bytes(image_bytes)
                rel = _posix_rel(out_dir, dest)
                figures.append(
                    ExtractedFigure(
                        page=page_num,
                        index_on_page=local_i,
                        filename=fname,
                        width=w,
                        height=h,
                        rel_posix=rel,
                    )
                )
    finally:
        doc.close()
    return figures


_FIGURE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def figure_stable_id(base_stem: str, page: int, idx: int) -> str:
    raw = f"{base_stem}_p{page}_i{idx}"
    return _FIGURE_ID_RE.sub("_", raw).strip("_")[:120] or f"fig_p{page}_i{idx}"


# Vision API: ограничить длинную сторону — у локальных VLM малый n_ctx на один запрос.
FIGURE_VISION_MAX_SIDE_PX = 1280
FIGURE_VISION_JPEG_QUALITY = 84


def _figure_image_data_url_for_api(image_path: Path, max_side: Optional[int] = None) -> str:
    """JPEG data URL после даунскейла; без PIL — сырой файл как раньше."""
    limit = max_side if max_side is not None else FIGURE_VISION_MAX_SIDE_PX
    try:
        from PIL import Image

        raw = image_path.read_bytes()
        img = Image.open(io.BytesIO(raw))
        img.load()
        w, h = img.size
        max_side = limit
        if max(w, h) > max_side:
            if w >= h:
                nh = max(1, int(round(h * (max_side / w))))
                nw = max_side
            else:
                nw = max(1, int(round(w * (max_side / h))))
                nh = max_side
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except AttributeError:
                resample = Image.LANCZOS
            img = img.resize((nw, nh), resample)
        img_rgb = img.convert("RGB") if img.mode != "RGB" else img
        bio = io.BytesIO()
        img_rgb.save(bio, format="JPEG", quality=FIGURE_VISION_JPEG_QUALITY, optimize=True)
        b64 = base64.standard_b64encode(bio.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        suf = image_path.suffix.lower()
        mime = "image/png"
        if suf in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suf == ".webp":
            mime = "image/webp"
        elif suf == ".gif":
            mime = "image/gif"
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"


VISION_SYSTEM = (
    "Ты смотришь на иллюстрацию из книги или PDF. "
    "Главное — передать смысл коротко и перенести читаемый текст с картинки и страницы как есть (подписи, подпись к рисунку, надписи на схеме). "
    "Не пытайся «нарисовать» схему словами: не перечисляй каждую стрелку и блок, если это не нужно для понимания. "
    "Пиши правдиво по видимому; нечитаемое помечай как неразборчиво."
)

VISION_USER_TEMPLATE = """На скриншоте — фрагмент или целая страница книги (номер страницы в издании: {page}).

Сформируй ответ по-русски, строго в таких блоках (с заголовками строк):

ЗАГОЛОВОК: одна короткая строка — о чём рисунок (до ~12 слов).

ОПИСАНИЕ: сначала 2–4 предложения своими словами: что за тип иллюстрации и о чём она по смыслу (не дотошный инвентарь «что где лежит»). Затем отдельным абзацем или списком выпиши весь разборчивый текст с изображения и с полей страницы рядом с ним: подпись под рисунком, заголовки, подписи к осям, легенда, заметные надписи на схеме — максимально дословно, как видишь. Не воспроизводи визуальную геометрию и не описывай каждый элемент «чтобы заменить картинку»; цель — текст + краткий смысл.

ГДЕ_ИСКАТЬ_В_КНИГЕ: 1–3 предложения — страница {page}, что обычно рядом в вёрстке, без выдуманных номеров разделов.

ДЛЯ_ИИ_СТАТЬИ: 2–4 предложения — как сослаться («как в книге, стр. {page}»), какую мысль несёт рисунок; напомни, что для точности лучше оригинальная иллюстрация, если нужна деталь схемы.
"""


def _is_vlm_context_overflow(exc: BaseException) -> bool:
    """Те же признаки, что и в file_processing (без циклического импорта)."""
    parts = [str(exc).lower()]
    m = getattr(exc, "message", None)
    if m:
        parts.append(str(m).lower())
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            parts.append(str(err["message"]).lower())
    blob = " ".join(parts)
    return any(
        x in blob
        for x in (
            "exceeds the available context",
            "context window",
            "context length",
            "requested exceeds",
            "token limit",
            "too many tokens",
            "prompt is too long",
            "n_ctx",
        )
    )


def describe_figure_with_openai_vision(
    client: Any,
    model: str,
    image_path: Path,
    page: int,
) -> str:
    """Один запрос vision; при переполнении контекста уменьшает картинку и повторяет."""
    max_side = FIGURE_VISION_MAX_SIDE_PX
    last_err: Optional[BaseException] = None
    for _ in range(10):
        try:
            url = _figure_image_data_url_for_api(image_path, max_side=max_side)
            resp = client.chat.completions.create(
                model=model or "gpt-4o",
                messages=[
                    {"role": "system", "content": VISION_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_USER_TEMPLATE.format(page=page)},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=900,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if _is_vlm_context_overflow(e) and max_side > 512:
                max_side = max(512, int(max_side * 0.55))
                continue
            raise
    if last_err:
        raise last_err
    return ""


def build_figures_markdown_section(
    base_stem: str,
    figures: list[ExtractedFigure],
    descriptions: Optional[dict[tuple[int, int], str]] = None,
) -> str:
    """
    Один раздел для вставки в конец основного .md или в отдельный файл при сплите.
    descriptions: ключ (page, index_on_page) -> текст от vision.
    """
    if not figures:
        return ""

    lines: list[str] = [
        "---",
        "",
        "## Иллюстрации и рисунки (извлечены из PDF)",
        "",
        "> **Для ИИ:** при написании статьи или конспекта по этому файлу используйте оригинальные "
        "рисунки из книги там, где это уместно. Ниже для каждого файла указаны **номер страницы в PDF**, "
        "**относительный путь к сохранённому файлу** рядом с markdown и **описание содержимого**. "
        "Вставляя иллюстрацию в публикацию, ссылайтесь на источник (книга, страница).",
        "",
    ]

    desc_map = descriptions or {}
    for fig in figures:
        fid = figure_stable_id(base_stem, fig.page, fig.index_on_page)
        lines.append(f"<!-- llmmd-figure id=\"{fid}\" pdf-page=\"{fig.page}\" file=\"{fig.rel_posix}\" -->")
        lines.append("")
        lines.append(f"### Рисунок — PDF стр. {fig.page}, файл `{fig.rel_posix}` ({fig.width}x{fig.height})")
        lines.append("")
        lines.append(f"![Иллюстрация (стр. {fig.page} PDF)]({fig.rel_posix})")
        lines.append("")
        key = (fig.page, fig.index_on_page)
        body = desc_map.get(key)
        if body:
            lines.append("**Автоописание (модель по картинке):**")
            lines.append("")
            lines.append(body)
            lines.append("")
        else:
            lines.append(
                "*Описание не запрашивалось или недоступно — ориентируйтесь по превью выше и номеру страницы в книге/PDF.*"
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_figure_descriptions(
    client: Any,
    model: str,
    figures: list[ExtractedFigure],
    out_dir: Path,
    cancel_event: Optional[threading.Event],
    log: Callable[[str], None],
    max_workers: int = 4,
) -> dict[tuple[int, int], str]:
    """
    Параллельно описывает рисунки через vision-модель.
    max_workers — максимальное число одновременных запросов к API.
    """
    out: dict[tuple[int, int], str] = {}
    if not figures:
        return out

    log_lock = threading.Lock()

    def _safe_log(msg: str) -> None:
        with log_lock:
            log(msg)

    def _describe_one(fig: ExtractedFigure) -> tuple[tuple[int, int], str]:
        if cancel_event is not None and cancel_event.is_set():
            return (fig.page, fig.index_on_page), ""
        path = out_dir / Path(fig.rel_posix)
        if not path.is_file():
            _safe_log(f"  описание рисунка: файл не найден {path}")
            return (fig.page, fig.index_on_page), ""
        try:
            _safe_log(f"  описание рисунка: стр. {fig.page}, {fig.filename}…")
            txt = describe_figure_with_openai_vision(client, model, path, fig.page)
            return (fig.page, fig.index_on_page), txt or ""
        except Exception as e:
            _safe_log(f"  описание рисунка {fig.filename}: {e}")
            return (fig.page, fig.index_on_page), f"*(ошибка модели: {e})*"

    n = len(figures)
    workers = max(1, min(max_workers, n))
    _safe_log(f"  описание рисунков: {n} шт., параллельно {workers} потока(ов)…")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_describe_one, fig): fig for fig in figures}
        for future in as_completed(futures):
            if cancel_event is not None and cancel_event.is_set():
                _safe_log("  иллюстрации: остановка — часть описаний пропущена.")
                for f in futures:
                    f.cancel()
                break
            key, txt = future.result()
            if txt:
                out[key] = txt

    done = len(out)
    _safe_log(f"  описание рисунков: готово {done}/{n}.")
    return out
