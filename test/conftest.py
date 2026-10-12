"""Shared test helpers. pytest loads this file automatically."""

from pathlib import Path

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


@pytest.fixture(scope="session")
def phase2_dirs(tmp_path_factory):
    """Phase 2 run ONCE into a temporary folder, so your real data/processed is never touched."""
    from src.ingest import IngestionSettings, run_ingestion

    if not Path("data/raw").exists():
        pytest.skip("Nexora data not found in data/raw")
    tmp = tmp_path_factory.mktemp("kora")
    run_ingestion(IngestionSettings(processed_dir=tmp / "processed", chunks_dir=tmp / "chunks"))
    return tmp


@pytest.fixture(scope="session")
def intel(phase2_dirs):
    """Phase 3 on the real Nexora data, run ONCE for the whole test session (nothing written)."""
    from src.build_intelligence import IntelligenceSettings, run_intelligence

    _, result = run_intelligence(IntelligenceSettings(
        processed_dir=phase2_dirs / "processed", chunks_dir=phase2_dirs / "chunks", save_output=False))
    return result
