"""Парсинг LLM TOC и эвристика оглавления (без вызова LLM)."""

from gui.services.file_processing import (
    _parse_llm_toc_json,
    infer_toc_sections_heuristic,
    split_by_toc_anchor_strings,
)


def test_parse_llm_toc_json_strict():
    raw = '{"sections": [{"anchor": "Глава 1", "title": "Введение"}]}'
    assert _parse_llm_toc_json(raw) == [("Глава 1", "Введение")]


def test_parse_llm_toc_json_after_thinking_block():
    raw = (
        "<think>Ищу заголовки</think>\n"
        '{"sections": [{"anchor": "Глава 2", "title": "Методы"}]}'
    )
    assert _parse_llm_toc_json(raw) == [("Глава 2", "Методы")]


def test_parse_llm_toc_json_inside_thinking_block():
    raw = (
        '<think>{"sections": [{"anchor": "Раздел A", "title": "Теория"}]}'
        "</think>"
    )
    assert _parse_llm_toc_json(raw) == [("Раздел A", "Теория")]


def test_parse_llm_toc_json_empty_sections():
    assert _parse_llm_toc_json('{"sections": []}') == []


def test_heuristic_finds_chapter_lines():
    text = "Введение\n\nГлава 1. Основы\n\nТекст главы.\n\nГлава 2. Практика\n\nЕщё текст."
    items = infer_toc_sections_heuristic(text)
    assert len(items) >= 2
    sections = split_by_toc_anchor_strings(text, items)
    assert len(sections) >= 2


def test_heuristic_finds_markdown_headings():
    text = "# Preface\n\nIntro.\n\n## Methods\n\nDetails.\n\n## Results\n\nData."
    items = infer_toc_sections_heuristic(text)
    assert len(items) >= 2
    sections = split_by_toc_anchor_strings(text, items)
    assert len(sections) >= 2
