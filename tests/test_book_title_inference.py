"""Парсинг и валидация ИИ-названия книги (без вызова LLM)."""

from gui.services.file_processing import (
    _book_title_is_descriptive_guess,
    _openai_assistant_text,
    _parse_llm_book_title_json,
    _strip_llm_thinking_blocks,
)


def test_parse_llm_book_title_json_strict():
    assert _parse_llm_book_title_json('{"title": "ГОСТ 7.0-2008"}') == "ГОСТ 7.0-2008"


def test_parse_llm_book_title_json_codeblock():
    raw = '```json\n{"title": "Системный анализ"}\n```'
    assert _parse_llm_book_title_json(raw) == "Системный анализ"


def test_parse_llm_book_title_json_embedded_prose():
    raw = 'Вот ответ: {"title": "Механика сплошных сред"} — готово.'
    assert _parse_llm_book_title_json(raw) == "Механика сплошных сред"


def test_parse_llm_book_title_json_truncated_object():
    raw = '{"title": "Теория упругости"'
    assert _parse_llm_book_title_json(raw) == "Теория упругости"


def test_parse_llm_book_title_json_plain_short_line():
    assert _parse_llm_book_title_json("Квантовая химия") == "Квантовая химия"


def test_descriptive_guess_rejects_summary():
    assert _book_title_is_descriptive_guess(
        "Как авторы выделяют основные понятия в данной главе"
    )
    assert _book_title_is_descriptive_guess(
        "Данная книга описывает методы системного анализа."
    )


def test_descriptive_guess_accepts_real_titles():
    assert not _book_title_is_descriptive_guess("ГОСТ Р 7.0.97-2016")
    assert not _book_title_is_descriptive_guess("Механика сплошных сред")
    assert not _book_title_is_descriptive_guess("Введение в теорию вероятностей")


def test_strip_llm_thinking_blocks():
    raw = (
        "<think>\nАнализирую титул...\n</think>\n"
        '{"title": "Системный анализ"}'
    )
    assert _strip_llm_thinking_blocks(raw) == '{"title": "Системный анализ"}'


def test_parse_llm_book_title_json_after_thinking_block():
    raw = (
        "<think>Думаю над названием</think>\n"
        '{"title": "Итог1"}'
    )
    assert _parse_llm_book_title_json(raw) == "Итог1"


def test_parse_llm_book_title_json_inside_thinking_block():
    raw = '<think>{"title": "Теория упругости"}</think>'
    assert _parse_llm_book_title_json(raw) == "Теория упругости"


def test_openai_assistant_text_reasoning_content_fallback():
    class _Msg:
        content = ""
        reasoning_content = (
            "<think>ok</think>\n{\"title\": \"Квантовая химия\"}"
        )

    assert _openai_assistant_text(_Msg()) == '{"title": "Квантовая химия"}'
