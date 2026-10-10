"""
KORA ingestion entry point.

One function with predictable input and output, for anything that wants to
run ingestion: a script, a test, or later the SkillKitLearn frontend.

    from src.ingest import IngestionSettings, run_ingestion
    summary, result = run_ingestion(IngestionSettings(include_archived=False))

From the terminal (KORA root):
    python -m src.ingest
    python -m src.ingest --exclude-archived --no-save
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src.chunking.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from src.pipeline import PipelineResult, process_folder
from src.storage.store import save_result
from src.validation.validator import Severity


@dataclass
class IngestionSettings:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chunks_dir: Path = Path("data/chunks")
    include_archived: bool = True     # default: employees see what changed and when
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_OVERLAP
    save_output: bool = True


@dataclass
class IngestionSummary:
    """A plain, JSON-friendly summary: what a frontend would display."""
    documents: int
    tables: int
    chunks: int
    relationships: int
    unresolved_references: int
    skipped_files: list[str]
    excluded_archived: list[str]
    errors: int
    warnings: int
    files_written: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def run_ingestion(
    settings: IngestionSettings | None = None,
) -> tuple[IngestionSummary, PipelineResult]:
    """Run the full pipeline with the given settings.

    Returns (summary, result): the summary for display, the full result
    for Python code that needs the details.
    """
    settings = settings or IngestionSettings()

    result = process_folder(
        settings.raw_dir,
        include_archived=settings.include_archived,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    written = []
    if settings.save_output:
        written = save_result(result, settings.processed_dir, settings.chunks_dir)

    severities = Counter(issue.severity for issue in result.validation_issues)

    summary = IngestionSummary(
        documents=len(result.documents),
        tables=len(result.tables),
        chunks=sum(len(doc.chunks) for doc in result.documents),
        relationships=len(result.relationships),
        unresolved_references=len(result.unresolved_references),
        skipped_files=result.skipped,
        excluded_archived=result.excluded,
        errors=severities[Severity.ERROR],
        warnings=severities[Severity.WARNING],
        files_written=[str(p) for p in written],
    )
    return summary, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KORA ingestion.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--exclude-archived", action="store_true",
                        help="Leave out documents with Status: Archived")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-save", action="store_true",
                        help="Run and report, but don't write output files")
    args = parser.parse_args()

    summary, result = run_ingestion(IngestionSettings(
        raw_dir=args.raw_dir,
        include_archived=not args.exclude_archived,
        chunk_size=args.chunk_size,
        save_output=not args.no_save,
    ))

    print(json.dumps(summary.to_dict(), indent=2))
    for issue in result.validation_issues:
        print(f"[{issue.severity.value.upper()}] {issue.message}  (subject: {issue.subject})")


if __name__ == "__main__":
    main()