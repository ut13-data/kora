"""
Stage 8b: Store

Writes a PipelineResult to disk:

  data/processed/documents.json          one metadata record per document
  data/processed/relationships.json      every derived relationship
  data/processed/tables/<name>.json      one file per CSV, rows kept as-is
  data/processed/validation_report.json  summary, skipped files, all issues
  data/chunks/chunks.jsonl               one chunk per line (JSON Lines)

Each run fully overwrites these files: the output always reflects
exactly what is currently in data/raw.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from src.pipeline import PipelineResult


def _to_jsonable(obj):
    """Translate types json can't handle on its own.

    json.dump calls this ONLY for objects it doesn't already understand.
    """
    if isinstance(obj, Enum):
        return obj.value              # FileType.MARKDOWN -> "markdown"
    if isinstance(obj, datetime):
        return obj.isoformat()        # datetime -> "2026-10-01T16:47:12+00:00"
    raise TypeError(f"Don't know how to store type {type(obj).__name__}")


def _write_json(data, path: Path) -> None:
    """Write data as pretty-printed JSON.

    encoding="utf-8" + ensure_ascii=False keeps characters like "€" and
    German umlauts readable in the file, instead of escaping them to "\\u20ac".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_to_jsonable)


def save_result(result: PipelineResult, processed_dir: Path, chunks_dir: Path) -> list[Path]:
    """Save every part of a PipelineResult to disk.

    Returns the list of files written, so callers (and later, a frontend)
    can show exactly what was produced.
    """
    written: list[Path] = []

    # ---- Documents: metadata only, plus a chunk count for quick inspection.
    # The chunks themselves live in chunks.jsonl, linked by document_id.
    documents = [
        {**asdict(doc.metadata), "chunk_count": len(doc.chunks)}
        for doc in result.documents
    ]
    path = processed_dir / "documents.json"
    _write_json(documents, path)
    written.append(path)

    # ---- Relationships: flattened across all documents.
    relationships = [asdict(rel) for rel in result.relationships]
    path = processed_dir / "relationships.json"
    _write_json(relationships, path)
    written.append(path)

    # ---- Tables: one file per CSV, so each keeps its own structure.
    # .stem is the filename without extension: "employees.csv" -> "employees"
    for table in result.tables:
        file_info = table.extracted.file_info
        path = processed_dir / "tables" / f"{Path(file_info.filename).stem}.json"
        _write_json({"file_info": asdict(file_info), "rows": table.extracted.rows}, path)
        written.append(path)

    # ---- Validation report: so problems are saved, not just printed.
    report = {
        "generated_at": datetime.now(timezone.utc),
        "summary": {
            "documents": len(result.documents),
            "tables": len(result.tables),
            "chunks": sum(len(doc.chunks) for doc in result.documents),
            "relationships": len(relationships),
            "skipped": len(result.skipped),
            "issues": len(result.validation_issues),
            "unresolved_references": len(result.unresolved_references),
        },
        "unresolved_references": result.unresolved_references,
        "skipped": result.skipped,
        "issues": [asdict(issue) for issue in result.validation_issues],
    }
    path = processed_dir / "validation_report.json"
    _write_json(report, path)
    written.append(path)

    # ---- Chunks: JSON Lines, one chunk per line.
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_dir / "chunks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for doc in result.documents:
            for chunk in doc.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False, default=_to_jsonable))
                f.write("\n")
    written.append(path)

    return written