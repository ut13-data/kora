"""Shared test helpers. pytest loads this file automatically."""

import pytest

from src.extraction.extractor import ExtractedDocument
from src.ingestion.file_intake import FileInfo, FileType


@pytest.fixture
def make_document():
    """Factory: build an in-memory ExtractedDocument from a string,
    so tests don't need real files on disk."""
    def _make(text: str, filename: str = "sample.md", file_type: FileType = FileType.MARKDOWN):
        info = FileInfo(filename=filename, file_type=file_type,
                        size_bytes=len(text), source_path=filename)
        return ExtractedDocument(file_info=info, text=text)
    return _make