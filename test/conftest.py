"""Shared test helpers. pytest loads this file automatically."""

import json
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


# ---------------- Phase 5: the saved graph, loaded the runtime way ----------------

@pytest.fixture(scope="session")
def built(phase2_dirs, tmp_path_factory):
    """Phase 5: Phase 3 saved to a temporary data/intelligence, then loaded the Phase 5 way."""
    from src.build_intelligence import IntelligenceSettings, run_intelligence
    from src.graph.load import load_graph
    from src.graph.trace import Tracer

    out = tmp_path_factory.mktemp("intel")
    run_intelligence(IntelligenceSettings(processed_dir=phase2_dirs / "processed",
                                          chunks_dir=phase2_dirs / "chunks", output_dir=out))
    loaded = load_graph(out)
    tracer = Tracer.from_dirs(loaded.graph, phase2_dirs / "processed", phase2_dirs / "chunks")
    tables = {p.stem: json.loads(p.read_text(encoding="utf-8"))["rows"]
              for p in (phase2_dirs / "processed" / "tables").glob("*.json")}
    return {"dir": out, "loaded": loaded, "graph": loaded.graph, "tracer": tracer, "tables": tables,
            "phase2": phase2_dirs}


@pytest.fixture(scope="session")
def users(built):
    from src.retrieval.access import users_from_tables
    return users_from_tables(built["tables"])


@pytest.fixture(scope="session")
def config():
    from src.retrieval.config import parse_config
    return parse_config({
        "access": {"rules": {"PROC-FIN-003": ["dept:DEPT-FIN", "dept:DEPT-OPS"],
                             "DOC-DEPT-FIN": ["dept:DEPT-FIN", "manager"]}},
        "admins": {"employees": ["EMP-0120"]},
        "person_visibility": {"colleague_fields": ["role_id", "role_title", "dept_id", "location_id", "manager_id"]},
    })
