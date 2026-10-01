"""
Pipeline Orchestrator

Runs a single file (or a whole folder) through every stage of the
ingestion pipeline: Upload -> Extract -> Clean -> Chunk -> Metadata.

Validation runs separately, across the WHOLE batch, since checks like
duplicate-ID detection are meaningless for a single file in isolation.
"""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.file_intake import identify_file, UnsupportedFileTypeError
from src.extraction.extractor import extract_file, ExtractedDocument, ExtractedTable, ExtractionError
from src.cleaning.cleaner import clean_extracted
from src.metadata.document_metadata import identify_document_metadata, DocumentMetadata
from src.chunking.chunker import chunk_document, Chunk
from src.metadata.relationships import derive_document_relationships, Relationship
from src.validation.validator import run_all_validations, ValidationIssue


@dataclass
class ProcessedDocument:
    """Everything the pipeline produced for ONE document file."""
    metadata: DocumentMetadata
    chunks: list[Chunk]
    relationships: list[Relationship]


@dataclass
class ProcessedTable:
    """A processed CSV file. Tables skip metadata/chunking/relationships --
    those stages are document-specific, as established back in Stage 4."""
    extracted: ExtractedTable


@dataclass
class PipelineResult:
    """The full output of running the pipeline over a batch of files."""
    documents: list[ProcessedDocument]
    tables: list[ProcessedTable]
    skipped: list[str]  # filenames skipped due to unsupported type or extraction failure
    validation_issues: list[ValidationIssue]


def process_file(file_path: Path) -> ProcessedDocument | ProcessedTable:
    """Run Stages 1-6 on a single file.

    Raises UnsupportedFileTypeError or ExtractionError if the file can't
    be processed -- callers deciding how to handle a batch (skip vs. stop)
    should catch these; this function doesn't swallow them.
    """
    info = identify_file(file_path)
    extracted = extract_file(info)
    cleaned = clean_extracted(extracted)

    if isinstance(cleaned, ExtractedTable):
        return ProcessedTable(extracted=cleaned)

    # isinstance(cleaned, ExtractedDocument)
    metadata = identify_document_metadata(cleaned)
    chunks = chunk_document(cleaned, metadata)
    relationships = derive_document_relationships(metadata)

    return ProcessedDocument(metadata=metadata, chunks=chunks, relationships=relationships)


def process_folder(folder: Path) -> PipelineResult:
    """Run the full pipeline (Stages 1-7) over every file in a folder.

    Files that fail Stage 1 or 2 are recorded in `skipped` rather than
    stopping the whole batch -- one bad file shouldn't block everything else.
    """
    documents: list[ProcessedDocument] = []
    tables: list[ProcessedTable] = []
    skipped: list[str] = []

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            result = process_file(file_path)
        except (UnsupportedFileTypeError, ExtractionError) as e:
            skipped.append(f"{file_path.name}: {e}")
            continue

        if isinstance(result, ProcessedDocument):
            documents.append(result)
        else:
            tables.append(result)

    all_metadata = [doc.metadata for doc in documents]
    all_chunks = [chunk for doc in documents for chunk in doc.chunks]
    all_relationships = [rel for doc in documents for rel in doc.relationships]

    validation_issues = run_all_validations(all_metadata, all_chunks, all_relationships)

    return PipelineResult(
        documents=documents,
        tables=tables,
        skipped=skipped,
        validation_issues=validation_issues,
    )