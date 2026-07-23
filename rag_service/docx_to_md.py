from pathlib import Path
from typing import List, Optional

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None


def convert_docx_to_markdown(docx_path: Path, output_dir: Path) -> Optional[Path]:
    """Convert a single DOCX file to Markdown using MarkItDown."""
    if MarkItDown is None:
        raise RuntimeError("markitdown package is not installed")

    md = MarkItDown()
    result = md.convert(str(docx_path))

    if not result or not result.text:
        return None

    output_path = output_dir / f"{docx_path.stem}.md"
    output_path.write_text(result.text, encoding="utf-8")

    return output_path


def convert_docx_folder_to_markdown(input_dir: Path, output_dir: Path) -> List[Path]:
    """Convert all DOCX files from input_dir to Markdown in output_dir."""
    if MarkItDown is None:
        raise RuntimeError("markitdown package is not installed")

    output_dir.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(input_dir.glob("*.docx"))
    converted = []

    for docx_path in docx_files:
        try:
            md_path = convert_docx_to_markdown(docx_path, output_dir)
            if md_path:
                converted.append(md_path)
        except Exception as e:
            print(f"Error converting {docx_path}: {e}")

    return converted


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
