"""Unit tests: Stage 1 (file intake) and Stage 3 (cleaning)."""

import pytest

from src.cleaning.cleaner import clean_row, clean_text
from src.ingestion.file_intake import FileType, UnsupportedFileTypeError, identify_file


def test_identifies_markdown_file(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("# Hello", encoding="utf-8")
    info = identify_file(f)
    assert info.filename == "guide.md"
    assert info.file_type == FileType.MARKDOWN
    assert info.size_bytes == 7


def test_uppercase_extension_is_accepted(tmp_path):
    f = tmp_path / "REPORT.PDF"
    f.write_bytes(b"%PDF-1.4")
    assert identify_file(f).file_type == FileType.PDF


def test_unsupported_type_is_rejected(tmp_path):
    f = tmp_path / "archive.zip"
    f.write_bytes(b"PK")
    with pytest.raises(UnsupportedFileTypeError):
        identify_file(f)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        identify_file(tmp_path / "nope.md")


def test_clean_text_collapses_blank_lines_and_tabs():
    assert clean_text("A\n\n\n\nB\t\t C") == "A\n\nB C"


def test_clean_text_never_changes_words_or_numbers():
    text = "Discounts above 30% need VP Sales approval within 10 working days."
    assert clean_text(text) == text


def test_clean_row_trims_cells_only():
    assert clean_row({"dept": " Sales ", "count": "5"}) == {"dept": "Sales", "count": "5"}