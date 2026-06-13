"""Тесты эвристик pdf_raster_profile (без тяжёлых PDF в репозитории)."""

from gui.services.pdf_raster_profile import (
    FRAGMENT_FLOOD_PER_PAGE,
    FULLPAGE_COVERAGE_MIN,
    PdfRasterProfile,
    _bbox_covers_page,
    _image_passes_size_filter,
    _sample_page_indices,
    page_should_use_render_ocr,
)


def test_bbox_covers_page():
    assert _bbox_covers_page(0, 0, 100, 100, 100, 100) is True
    assert _bbox_covers_page(0, 0, 10, 10, 100, 100) is False


def test_image_size_filter():
    assert _image_passes_size_filter(100, 100) is True
    assert _image_passes_size_filter(10, 10) is False


def test_page_should_use_render_ocr():
    assert page_should_use_render_ocr(FRAGMENT_FLOOD_PER_PAGE) is False
    assert page_should_use_render_ocr(FRAGMENT_FLOOD_PER_PAGE + 1) is True


def test_fullpage_ratio_constant():
    assert FULLPAGE_COVERAGE_MIN > 0.5


def test_sample_page_indices():
    sampled = _sample_page_indices(list(range(100)))
    assert len(sampled) == 7
    assert sampled[0] == 0 and sampled[-1] == 99
    assert _sample_page_indices([0, 1, 2]) == [0, 1, 2]


def test_fragment_flood_skips_extract():
    p = PdfRasterProfile(
        kind="fragment_flood",
        page_count=10,
        images_total=5000,
        max_images_on_page=409,
        fullpage_scan_pages=0,
        has_text_layer=False,
        message="test",
    )
    assert p.skip_figure_extract is True
    assert p.use_page_render_ocr is True


def test_scanned_fullpage_still_extracts_assets():
    p = PdfRasterProfile(
        kind="scanned_fullpage",
        page_count=120,
        images_total=120,
        max_images_on_page=1,
        fullpage_scan_pages=120,
        has_text_layer=False,
        message="test",
    )
    assert p.skip_figure_extract is False
    assert p.use_page_render_ocr is True
