from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Согласовано с services/file_processing.split_markdown_by_heading (уровень + шумные заголовки).

_NOISE_MD_HEADING = re.compile(
    r"^(page|стр\.?|страница|лист\.?)\s*:?\s*\d+\s*$",
    re.IGNORECASE,
)


def _is_noise_markdown_heading(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    if _NOISE_MD_HEADING.match(t):
        return True
    if re.fullmatch(r"\d{1,4}", t):
        return True
    return False


def split_markdown_by_heading_level(
    text: str,
    level: int = 2,
    *,
    skip_noise_headings: bool = True,
) -> list[tuple[str, str]]:
    """
    Те же границы секций, что в GUI MarkItDown при сплите по H1/H2/H3:
    только строки с ровно `level` решёток, служебные заголовки пропускаются.
    """
    if level < 1 or level > 6:
        raise ValueError("level must be 1..6")
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
        chunk = text[last_pos : match.start()].strip()
        if chunk or last_title != "_preface":
            sections.append((last_title, chunk))
        last_pos = match.start()
        last_title = title

    chunk = text[last_pos:].strip()
    if chunk:
        sections.append((last_title, chunk))

    return sections


def _split_oversized(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or len(text) <= size:
        return [text] if text else []
    out: list[str] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(text):
        piece = text[i : i + size]
        if piece.strip():
            out.append(piece)
        i += step
    return out


@dataclass(frozen=True)
class TextChunk:
    text: str
    heading: str
    chunk_index: int


def chunk_markdown_file(
    body: str,
    *,
    heading_level: int = 2,
    chunk_max_chars: int = 0,
    chunk_overlap_chars: int = 0,
) -> list[TextChunk]:
    """
    По умолчанию — только секции по заголовку заданного уровня (как после сплита в основном приложении).
    Уже разбитые на файлы `001_....md` обычно дают одну секцию на документ.

    Если chunk_max_chars > 0 — дополнительная донарезка только слишком длинных секций (запасной режим).
    """
    raw = split_markdown_by_heading_level(body, level=heading_level)
    out: list[TextChunk] = []
    idx = 0
    for heading, content in raw:
        if not content.strip():
            continue
        if chunk_max_chars > 0:
            pieces = _split_oversized(content, chunk_max_chars, chunk_overlap_chars)
        else:
            pieces = [content]
        for piece in pieces:
            t = piece.strip()
            if not t:
                continue
            out.append(TextChunk(text=t, heading=heading, chunk_index=idx))
            idx += 1
    return out


def iter_markdown_files(root: Path, *, glob_pattern: str = "**/*.md") -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob(glob_pattern) if p.is_file())


def find_or_convert_docx_files(root: Path, glob_pattern: str = "**/*.docx") -> list[Path]:
    """
    Find .md files, or convert .docx files to .md if .md not found.
    Returns list of .md files to index.
    """
    md_files = sorted(root.glob("**/*.md"))

    if md_files:
        return md_files

    try:
        from .docx_to_md import convert_docx_folder_to_markdown

        output_dir = root / "outputs"
        converted = convert_docx_folder_to_markdown(root, output_dir)

        return converted
    except Exception:
        return []
