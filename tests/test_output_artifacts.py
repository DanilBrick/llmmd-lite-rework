"""Имена выходных артефактов и отчёты OCR-пайплайна."""

import queue
from pathlib import Path

from gui.services.duration_fmt import format_duration_ru
from gui.services.file_processing import save_split_sections, unique_output_stem, write_book_report
from gui.services.file_processing import default_enabled_stages, normalize_enabled_stages


def test_unique_output_stem_avoids_markdown_assets_split_and_report_collisions(tmp_path: Path):
    (tmp_path / "Book.md").write_text("old", encoding="utf-8")
    (tmp_path / "Book_assets").mkdir()
    (tmp_path / "Book_1").mkdir()
    (tmp_path / "Book_2_report.md").write_text("old report", encoding="utf-8")

    assert unique_output_stem(tmp_path, "Book") == "Book_3"


def test_unique_output_stem_keeps_clean_name_when_no_artifacts_exist(tmp_path: Path):
    assert unique_output_stem(tmp_path, "Book") == "Book"


def test_format_duration_ru_uses_hours_minutes_seconds():
    assert format_duration_ru(59) == "59 с"
    assert format_duration_ru(125) == "2 мин 5 с"
    assert format_duration_ru(7_321) == "2 ч 2 мин 1 с"


def test_write_book_report_includes_outputs_and_stage_durations(tmp_path: Path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF")
    output = tmp_path / "Book.md"
    output.write_text("# Book", encoding="utf-8")

    report = write_book_report(
        tmp_path,
        "Book",
        source=source,
        title_stem="Book",
        stem_how="ИИ",
        outputs=[output],
        stages=[
            {
                "stage": "convert",
                "label": "Конвертация/OCR в Markdown",
                "status": "done",
                "details": "весь документ",
                "duration_s": 3_661,
            }
        ],
        figures_count=2,
        sections_count=4,
        stopped=False,
    )

    text = report.read_text(encoding="utf-8")
    assert "`Book.md`" in text
    assert "Конвертация/OCR в Markdown: done, 1 ч 1 мин 1 с" in text
    assert "Извлечено изображений: 2" in text


def test_normalize_enabled_stages_filters_unknown_values():
    assert normalize_enabled_stages(["convert", "unknown", "split"]) == {"convert", "split"}


def test_normalize_enabled_stages_defaults_for_empty_or_invalid_values():
    assert normalize_enabled_stages([]) == set(default_enabled_stages())
    assert normalize_enabled_stages({"unknown"}) == set(default_enabled_stages())


def test_save_split_sections_can_create_obsidian_links(tmp_path: Path):
    events: queue.Queue = queue.Queue()

    saved = save_split_sections(
        [("Введение", "Текст 1"), ("Глава 1", "Текст 2")],
        tmp_path,
        "Book",
        events,
        obsidian_links=True,
    )

    assert saved == 2
    index = (tmp_path / "Book" / "000_index.md").read_text(encoding="utf-8")
    first = (tmp_path / "Book" / "001_Введение.md").read_text(encoding="utf-8")
    second = (tmp_path / "Book" / "002_Глава_1.md").read_text(encoding="utf-8")
    assert "[[001_Введение|Введение]]" in index
    assert 'llmmd_book: "Book"' in first
    assert "[[002_Глава_1|следующий раздел →]]" in first
    assert "[[001_Введение|← предыдущий раздел]]" in second
