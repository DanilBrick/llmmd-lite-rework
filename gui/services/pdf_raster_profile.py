"""Эвристики для PDF со скан-страницами (EBS) и «image flood» (сотни плиток на страницу)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Сколько встроенных image/xobject на странице считаем «лавиной» → один OCR всей страницы.
FRAGMENT_FLOOD_PER_PAGE = 20
# Не извлекать в *_assets/ и не описывать vision, если картинок больше (защита от 10k+).
MAX_FIGURES_EXTRACT_TOTAL = 500
# Доля страниц с одним растром на весь лист → режим scanned_fullpage.
SCANNED_FULLPAGE_PAGE_RATIO = 0.85
# Минимальная доля площади страницы, которую занимает растр (по bbox в пунктах PDF).
FULLPAGE_COVERAGE_MIN = 0.82
# Сколько страниц смотреть при probe (без декодирования каждого xref).
PROBE_MAX_SAMPLE_PAGES = 7
# Те же пороги, что в pdf_images.extract_figures_from_pdf
_MIN_WIDTH = 48
_MIN_HEIGHT = 48
_MIN_PIXELS = 4000


@dataclass(frozen=True)
class PdfRasterProfile:
    """Результат быстрого анализа PDF перед OCR / извлечением рисунков."""

    kind: str  # scanned_fullpage | fragment_flood | normal
    page_count: int
    images_total: int
    max_images_on_page: int
    fullpage_scan_pages: int
    has_text_layer: bool
    message: str

    @property
    def skip_figure_extract(self) -> bool:
        # Плитки (сотни xobject на страницу) — не извлекать. Скан-книга: один растр на лист → в assets.
        return self.kind == "fragment_flood"

    @property
    def use_page_render_ocr(self) -> bool:
        """Один OCR по рендеру страницы вместо сотен xobject / плиток в PDF."""
        return self.kind in ("scanned_fullpage", "fragment_flood")


def _image_passes_size_filter(width: int, height: int) -> bool:
    return width >= _MIN_WIDTH and height >= _MIN_HEIGHT and width * height >= _MIN_PIXELS


def _bbox_covers_page(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_w: float,
    page_h: float,
) -> bool:
    if page_w <= 0 or page_h <= 0:
        return False
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    return (bw * bh) / (page_w * page_h) >= FULLPAGE_COVERAGE_MIN


def _sample_page_indices(page_indices: list[int], max_samples: int = PROBE_MAX_SAMPLE_PAGES) -> list[int]:
    n = len(page_indices)
    if n <= max_samples:
        return list(page_indices)
    if max_samples <= 1:
        return [page_indices[0]]
    out: list[int] = []
    for i in range(max_samples):
        pos = int(round(i * (n - 1) / (max_samples - 1)))
        idx = page_indices[pos]
        if idx not in out:
            out.append(idx)
    return out


def _unique_xref_count_on_page(page: Any) -> int:
    """Число уникальных image xref на странице без extract_image (быстро)."""
    seen: set[int] = set()
    for info in page.get_images(full=True) or []:
        seen.add(int(info[0]))
    return len(seen)


def _page_fullpage_scan_heuristic(page: Any, unique_xrefs: int) -> bool:
    """Один крупный растр на весь лист — по bbox, без декодирования пикселей."""
    if unique_xrefs != 1:
        return False
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    for info in page.get_images(full=True) or []:
        xref = int(info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for r in rects:
            if _bbox_covers_page(r.x0, r.y0, r.x1, r.y1, pw, ph):
                return True
    text = (page.get_text() or "").strip()
    return not text


def probe_pdf_raster_profile(
    pdf_path: Path,
    pages_filter: Optional[set[int]] = None,
) -> PdfRasterProfile:
    """
    Быстрый анализ PDF (PyMuPDF): только подсчёт xref и bbox, без extract_image на всех плитках.
    """
    import fitz

    path = Path(pdf_path)
    doc = fitz.open(str(path))
    try:
        page_indices = list(range(doc.page_count))
        if pages_filter is not None:
            page_indices = [i for i in page_indices if (i + 1) in pages_filter]
        n = len(page_indices)
        if not n:
            return PdfRasterProfile(
                kind="normal",
                page_count=0,
                images_total=0,
                max_images_on_page=0,
                fullpage_scan_pages=0,
                has_text_layer=False,
                message="нет страниц для анализа",
            )

        sample = _sample_page_indices(page_indices)
        per_page_counts: list[int] = []
        fullpage_in_sample = 0
        has_text = False

        for page_idx in sample:
            page = doc[page_idx]
            if len((page.get_text() or "").strip()) > 40:
                has_text = True
            cnt = _unique_xref_count_on_page(page)
            per_page_counts.append(cnt)
            if _page_fullpage_scan_heuristic(page, cnt):
                fullpage_in_sample += 1

        max_on_page = max(per_page_counts) if per_page_counts else 0
        avg_on_page = sum(per_page_counts) / len(per_page_counts) if per_page_counts else 0.0
        images_total_est = int(round(avg_on_page * n))

        if max_on_page > FRAGMENT_FLOOD_PER_PAGE:
            kind = "fragment_flood"
            msg = (
                f"PDF с плитками: до {max_on_page} встроенных растров на стр. "
                f"(~{images_total_est:,} в документе) — OCR и извлечение по одной странице, без {max_on_page}xN вызовов"
            )
            return PdfRasterProfile(
                kind=kind,
                page_count=n,
                images_total=images_total_est,
                max_images_on_page=max_on_page,
                fullpage_scan_pages=0,
                has_text_layer=has_text,
                message=msg,
            )

        ratio = fullpage_in_sample / len(sample) if sample else 0.0
        if ratio >= SCANNED_FULLPAGE_PAGE_RATIO and max_on_page <= 3:
            kind = "scanned_fullpage"
            msg = (
                f"скан-книга (~{fullpage_in_sample}/{len(sample)} проверенных стр. — один растр на лист); "
                "страницы сохраняются в …_assets/"
            )
            return PdfRasterProfile(
                kind=kind,
                page_count=n,
                images_total=images_total_est,
                max_images_on_page=max_on_page,
                fullpage_scan_pages=int(round(ratio * n)),
                has_text_layer=has_text,
                message=msg,
            )

        # Обычный PDF: точный подсчёт только если мало xref на странице
        images_total = 0
        fullpage_pages = 0
        if max_on_page <= FRAGMENT_FLOOD_PER_PAGE:
            for page_idx in page_indices:
                page = doc[page_idx]
                if len((page.get_text() or "").strip()) > 40:
                    has_text = True
                cnt = _unique_xref_count_on_page(page)
                images_total += cnt
                if _page_fullpage_scan_heuristic(page, cnt):
                    fullpage_pages += 1

        return PdfRasterProfile(
            kind="normal",
            page_count=n,
            images_total=images_total,
            max_images_on_page=max_on_page,
            fullpage_scan_pages=fullpage_pages,
            has_text_layer=has_text,
            message=f"обычный PDF: {images_total} встроенных растров на {n} стр.",
        )
    finally:
        doc.close()


def page_should_use_render_ocr(images_on_page_count: int) -> bool:
    """Слишком много xobject на странице (pdfplumber) — один OCR по рендеру."""
    return images_on_page_count > FRAGMENT_FLOOD_PER_PAGE


# Активный профиль текущего job (устанавливается из file_processing перед convert).
_active_profile: Optional[PdfRasterProfile] = None


def set_active_pdf_raster_profile(profile: Optional[PdfRasterProfile]) -> None:
    global _active_profile
    _active_profile = profile


def get_active_pdf_raster_profile() -> Optional[PdfRasterProfile]:
    return _active_profile
