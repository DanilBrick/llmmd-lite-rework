"""Патчи MarkItDown-OCR: отмена по страницам PDF и логирование вызовов OCR в очередь событий."""

from __future__ import annotations

import inspect
import io
import queue
import threading
from typing import Any, BinaryIO, Optional

try:
    from markitdown import DocumentConverterResult, StreamInfo
    from markitdown._exceptions import MISSING_DEPENDENCY_MESSAGE, MissingDependencyException
except Exception:
    DocumentConverterResult = None
    StreamInfo = Any
    MISSING_DEPENDENCY_MESSAGE = ""
    MissingDependencyException = RuntimeError

from gui.services.pdf_raster_profile import (
    get_active_pdf_raster_profile,
    page_should_use_render_ocr,
)

OCR_LOG_TEXT_MAX = 8000

# Меньше DPI при полностраничном OCR + уменшение сторон изображений перед вызовом vision LLM —
# меньше визуальных токенов и нагрузка на маленький n_ctx локального сервера.
OCR_FULLPAGE_RENDER_RESOLUTION_DPI = 180
OCR_EMBEDDED_BITMAP_MAX_SIDE_PX = 1280

_ocr_logging_installed = False
_ocr_downscale_installed = False
_pdf_cancel_hooks_installed = False
_pdf_cancel_event: Optional[threading.Event] = None
_pdf_pause_event: Optional[threading.Event] = None


def set_pdf_cancel_event(ev: Optional[threading.Event]) -> None:
    global _pdf_cancel_event
    _pdf_cancel_event = ev


def set_pdf_pause_event(ev: Optional[threading.Event]) -> None:
    global _pdf_pause_event
    _pdf_pause_event = ev


def pdf_stop_requested() -> bool:
    return _pdf_cancel_event is not None and _pdf_cancel_event.is_set()


def pdf_wait_if_paused() -> None:
    while (
        _pdf_pause_event is not None
        and _pdf_pause_event.is_set()
        and not pdf_stop_requested()
    ):
        _pdf_pause_event.wait(0.25)


def _page_png_stream_for_ocr(page: Any, dpi: int = OCR_FULLPAGE_RENDER_RESOLUTION_DPI) -> io.BytesIO:
    """Рендер страницы PDF в PNG для одного vision-OCR (скан / image flood)."""
    import fitz

    sc = dpi / 72.0
    mat = fitz.Matrix(sc, sc)
    pix = page.get_pixmap(matrix=mat)
    bio = io.BytesIO(pix.tobytes("png"))
    bio.seek(0)
    return bio


def _append_page_render_ocr(
    markdown_content: list[str],
    page: Any,
    page_num: int,
    ocr_service: Any,
) -> None:
    """Один OCR на всю страницу вместо сотен встроенных плиток."""
    try:
        img_stream = _page_png_stream_for_ocr(page)
        ocr_result = ocr_service.extract_text(img_stream)
        if ocr_result.text.strip():
            markdown_content.append(f"*[Image OCR]\n{ocr_result.text.strip()}\n[End OCR]*")
        else:
            markdown_content.append("*[No text could be extracted from this page]*")
    except Exception as e:
        markdown_content.append(f"*[Error processing page {page_num}: {str(e)}]*")


def _downscale_png_stream(image_stream: BinaryIO | io.BytesIO, max_side_px: int) -> io.BytesIO:
    """
    Если PIL доступен и изображение крупнее max_side_px, уменьшает сохраняя пропорции (PNG).
    Иначе возвращает копию исходного потока, сброшенного в начало.
    """
    try:
        from PIL import Image
    except Exception:
        image_stream.seek(0)
        return io.BytesIO(image_stream.read())

    image_stream.seek(0)
    raw = image_stream.read()
    if not raw:
        return io.BytesIO()

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return io.BytesIO(raw)

    w, h = img.size
    if w <= max_side_px and h <= max_side_px:
        return io.BytesIO(raw)

    if w >= h:
        nh = max(1, int(round(h * (max_side_px / w))))
        nw = max_side_px
    else:
        nw = max(1, int(round(w * (max_side_px / h))))
        nh = max_side_px

    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:
        resample = Image.LANCZOS

    resized = img.resize((nw, nh), resample)
    bio = io.BytesIO()
    if resized.mode not in ("L", "RGB"):
        resized = resized.convert("RGB")
    resized.save(bio, format="PNG", optimize=True)
    bio.seek(0)
    return bio


def install_ocr_image_limits(max_side_px: int = OCR_EMBEDDED_BITMAP_MAX_SIDE_PX) -> None:
    """Ограничить разрешение каждого снимка до vision-OCR без правки цепочки convert (внутри extract_text)."""
    global _ocr_downscale_installed
    if _ocr_downscale_installed:
        return
    try:
        from markitdown_ocr._ocr_service import LLMVisionOCRService
    except ImportError:
        return

    _orig = LLMVisionOCRService.extract_text

    def extract_text_bounded(self, image_stream, prompt=None, stream_info=None, **kwargs):
        try:
            small = _downscale_png_stream(image_stream, max_side_px)
        except Exception:
            image_stream.seek(0)
            return _orig(self, image_stream, prompt, stream_info, **kwargs)
        return _orig(self, small, prompt, stream_info, **kwargs)

    LLMVisionOCRService.extract_text = extract_text_bounded
    _ocr_downscale_installed = True


def _infer_ocr_context_from_stack() -> str:
    for fr in inspect.stack()[2:22]:
        loc = fr.frame.f_locals
        if loc.get("img_info") and isinstance(loc["img_info"], dict):
            name = loc["img_info"].get("name") or "?"
            pn = loc.get("page_num")
            if pn is not None:
                return f"PDF стр. {pn}, встроенное изображение «{name}»"
            return f"PDF, изображение «{name}»"
        if "page_num" in loc:
            return f"PDF стр. {loc['page_num']}"
        if "slide_num" in loc:
            return f"PPTX, слайд {loc['slide_num']}"
        if "cell_ref" in loc and "sheet" in loc:
            cr = loc.get("cell_ref") or "?"
            sh = getattr(loc.get("sheet"), "title", None) or "?"
            return f"XLSX, лист «{sh}», ячейка {cr}"
        rel = loc.get("rel")
        if rel is not None and hasattr(rel, "target_ref"):
            tr = (getattr(rel, "target_ref", None) or "").lower()
            if "image" in tr or "media" in tr:
                return f"DOCX, вложение ({getattr(rel, 'target_ref', '?')})"
    return "OCR (контекст не определён)"


def install_ocr_gui_logging(events_queue: queue.Queue) -> None:
    global _ocr_logging_installed
    if _ocr_logging_installed:
        return
    try:
        from markitdown_ocr._ocr_service import LLMVisionOCRService
    except ImportError:
        return

    _orig = LLMVisionOCRService.extract_text

    def extract_text_logged(self, image_stream, prompt=None, stream_info=None, **kwargs):
        result = _orig(self, image_stream, prompt, stream_info, **kwargs)
        place = _infer_ocr_context_from_stack()
        n = len(result.text or "")
        head = "─" * 58
        lines = [head, f"OCR · {place}",
                 f"бэкенд: {result.backend_used or '—'} · символов в ответе: {n}", head]
        if result.error:
            lines.append(f"ошибка API/модели: {result.error}")
        body = (result.text or "").strip()
        if body:
            if len(body) > OCR_LOG_TEXT_MAX:
                lines.append(body[:OCR_LOG_TEXT_MAX])
                lines.append(f"... [лог обрезан: всего {len(body)} символов]")
            else:
                lines.append(body)
        else:
            lines.append("(пустой ответ модели)")
        lines.append(head)
        events_queue.put(("log", "\n".join(lines)))
        events_queue.put(("ocr_page", f"Последний OCR: {place}"))
        return result

    LLMVisionOCRService.extract_text = extract_text_logged
    _ocr_logging_installed = True


def _patched_ocr_full_pages(self, pdf_bytes: io.BytesIO, ocr_service: Any) -> str:
    import pdfplumber

    markdown_parts: list[str] = []
    try:
        pdf_bytes.seek(0)
        with pdfplumber.open(pdf_bytes) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                pdf_wait_if_paused()
                if pdf_stop_requested():
                    markdown_parts.append(
                        "\n*--- Остановка с сохранением: дальнейшие страницы не отправлялись в OCR ---*\n"
                    )
                    break
                try:
                    markdown_parts.append(f"\n## Page {page_num}\n")
                    page_img = page.to_image(resolution=OCR_FULLPAGE_RENDER_RESOLUTION_DPI)
                    img_stream = io.BytesIO()
                    page_img.original.save(img_stream, format="PNG")
                    img_stream.seek(0)
                    ocr_result = ocr_service.extract_text(img_stream)
                    if ocr_result.text.strip():
                        markdown_parts.append(f"*[Image OCR]\n{ocr_result.text.strip()}\n[End OCR]*")
                    else:
                        markdown_parts.append("*[No text could be extracted from this page]*")
                except Exception as e:
                    markdown_parts.append(f"*[Error processing page {page_num}: {str(e)}]*")
    except Exception:
        markdown_parts = []
        try:
            import fitz
            pdf_bytes.seek(0)
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            try:
                for page_num in range(1, doc.page_count + 1):
                    pdf_wait_if_paused()
                    if pdf_stop_requested():
                        markdown_parts.append(
                            "\n*--- Остановка с сохранением: дальнейшие страницы не отправлялись в OCR ---*\n"
                        )
                        break
                    try:
                        markdown_parts.append(f"\n## Page {page_num}\n")
                        page = doc[page_num - 1]
                        sc = OCR_FULLPAGE_RENDER_RESOLUTION_DPI / 72.0
                        mat = fitz.Matrix(sc, sc)
                        pix = page.get_pixmap(matrix=mat)
                        img_stream = io.BytesIO(pix.tobytes("png"))
                        img_stream.seek(0)
                        ocr_result = ocr_service.extract_text(img_stream)
                        if ocr_result.text.strip():
                            markdown_parts.append(f"*[Image OCR]\n{ocr_result.text.strip()}\n[End OCR]*")
                        else:
                            markdown_parts.append("*[No text could be extracted from this page]*")
                    except Exception as e:
                        markdown_parts.append(f"*[Error processing page {page_num}: {str(e)}]*")
            finally:
                doc.close()
        except Exception:
            return "*[Error: Could not process scanned PDF]*"

    return "\n\n".join(markdown_parts).strip()


def _patched_pdf_convert(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> DocumentConverterResult:
    from markitdown_ocr._pdf_converter_with_ocr import _dependency_exc_info
    if _dependency_exc_info is not None:
        raise MissingDependencyException(
            MISSING_DEPENDENCY_MESSAGE.format(
                converter=type(self).__name__, extension=".pdf", feature="pdf",
            )
        ) from _dependency_exc_info[1].with_traceback(_dependency_exc_info[2])

    ocr_service: Any = kwargs.get("ocr_service") or self.ocr_service
    file_stream.seek(0)
    pdf_bytes = io.BytesIO(file_stream.read())
    markdown_content: list[str] = []

    import pdfminer.high_level
    import pdfplumber

    fitz_doc: Any = None
    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                pdf_wait_if_paused()
                if pdf_stop_requested():
                    markdown_content.append(
                        "\n\n*--- Остановка с сохранением: следующие страницы пропущены ---*\n"
                    )
                    break
                markdown_content.append(f"\n## Page {page_num}\n")
                if ocr_service:
                    raster_profile = get_active_pdf_raster_profile()
                    use_render = raster_profile is not None and raster_profile.use_page_render_ocr
                    images_on_page = (
                        [] if use_render else self._extract_page_images(pdf_bytes, page_num)
                    )
                    if use_render or (
                        images_on_page and page_should_use_render_ocr(len(images_on_page))
                    ):
                        import fitz

                        if fitz_doc is None:
                            pdf_bytes.seek(0)
                            fitz_doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
                        _append_page_render_ocr(
                            markdown_content,
                            fitz_doc[page_num - 1],
                            page_num,
                            ocr_service,
                        )
                        continue
                    if images_on_page:
                        chars = page.chars
                        if chars:
                            lines_with_y = []
                            current_line = []
                            current_y = None
                            for char in sorted(chars, key=lambda c: (c["top"], c["x0"])):
                                y = char["top"]
                                if current_y is None:
                                    current_y = y
                                elif abs(y - current_y) > 2:
                                    if current_line:
                                        text = "".join([c["text"] for c in current_line])
                                        lines_with_y.append({"y": current_y, "text": text.strip()})
                                    current_line = []
                                    current_y = y
                                current_line.append(char)
                            if current_line:
                                text = "".join([c["text"] for c in current_line])
                                lines_with_y.append({"y": current_y, "text": text.strip()})
                        else:
                            text_content = page.extract_text() or ""
                            lines_with_y = [
                                {"y": i * 10, "text": line}
                                for i, line in enumerate(text_content.split("\n"))
                            ]
                        image_data = []
                        for img_info in images_on_page:
                            ocr_result = ocr_service.extract_text(img_info["stream"])
                            if ocr_result.text.strip():
                                image_data.append({
                                    "y_pos": img_info["y_pos"], "name": img_info["name"],
                                    "ocr_text": ocr_result.text, "backend": ocr_result.backend_used,
                                    "type": "image",
                                })
                        content_items = [
                            {"y_pos": item["y"], "text": item["text"], "type": "text"}
                            for item in lines_with_y if item["text"]
                        ]
                        content_items.extend(image_data)
                        content_items.sort(key=lambda x: x["y_pos"])
                        for item in content_items:
                            if item["type"] == "text":
                                markdown_content.append(item["text"])
                            else:
                                markdown_content.append(
                                    f"\n\n*[Image OCR]\n{item['ocr_text']}\n[End OCR]*\n"
                                )
                    else:
                        text_content = page.extract_text() or ""
                        if text_content.strip():
                            markdown_content.append(text_content.strip())
                else:
                    text_content = page.extract_text() or ""
                    if text_content.strip():
                        markdown_content.append(text_content.strip())

            markdown = "\n\n".join(markdown_content).strip()
            if not markdown:
                pdf_bytes.seek(0)
                markdown = pdfminer.high_level.extract_text(pdf_bytes)
    except Exception:
        try:
            pdf_bytes.seek(0)
            markdown = pdfminer.high_level.extract_text(pdf_bytes)
        except Exception:
            markdown = ""
    finally:
        if fitz_doc is not None:
            fitz_doc.close()

    if ocr_service and (not markdown or not markdown.strip()):
        pdf_bytes.seek(0)
        markdown = self._ocr_full_pages(pdf_bytes, ocr_service)

    return DocumentConverterResult(markdown=markdown)


def install_pdf_cancel_hooks() -> None:
    global _pdf_cancel_hooks_installed
    if _pdf_cancel_hooks_installed:
        return
    try:
        from markitdown_ocr._pdf_converter_with_ocr import PdfConverterWithOCR
    except ImportError:
        return
    PdfConverterWithOCR.convert = _patched_pdf_convert
    PdfConverterWithOCR._ocr_full_pages = _patched_ocr_full_pages
    _pdf_cancel_hooks_installed = True
