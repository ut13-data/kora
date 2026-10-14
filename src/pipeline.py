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
from src.metadata.document_metadata import (
    identify_document_metadata, apply_fallback_identity, DocumentMetadata,
)
from src.chunking.chunker import chunk_document, Chunk
from src.metadata.relationships import derive_document_relationships, Relationship
from src.validation.validator import run_all_validations, ValidationIssue
from src.metadata.discovery import discover_relationships
from src.metadata.location_links import derive_location_links
from src.chunking.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


from src.metadata.relationships import (
    derive_document_relationships, derive_table_relationships, Relationship,
)
from src.metadata.registry import build_registry, EntityRegistry


@dataclass
class ProcessedDocument:
    """Everything the pipeline produced for ONE document file."""
    metadata: DocumentMetadata
    chunks: list[Chunk]


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
    relationships: list[Relationship]
    unresolved_references: list[str]
    registry: EntityRegistry
    skipped: list[str]
    validation_issues: list[ValidationIssue]
    excluded: list[str]   # documents left out by settings (e.g. archived)


def process_file(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    root: Path | None = None,
) -> ProcessedDocument | ProcessedTable:
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
    # Documents without a header ID/title (real-world PDFs, DOCX...) get
    # labelled stand-ins, so two such files never share one identity.
    relative = file_path.relative_to(root) if root and file_path.is_relative_to(root) else Path(file_path.name)
    metadata = apply_fallback_identity(metadata, cleaned, relative.as_posix())
    chunks = chunk_document(cleaned, metadata, chunk_size=chunk_size, overlap=overlap)
    #relationships = derive_document_relationships(metadata)

    return ProcessedDocument(metadata=metadata, chunks=chunks)

def process_folder(
    folder: Path,
    include_archived: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> PipelineResult:
    """Run the full pipeline (Stages 1-7) over every file in a folder.

    Files that fail Stage 1 or 2 are recorded in `skipped` rather than
    stopping the whole batch -- one bad file shouldn't block everything else.
    """
    documents: list[ProcessedDocument] = []
    tables: list[ProcessedTable] = []
    skipped: list[str] = []
    excluded: list[str] = []
    
    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            result = process_file(file_path, chunk_size, overlap, root=folder)
        except (UnsupportedFileTypeError, ExtractionError) as e:
            skipped.append(f"{file_path.name}: {e}")
            continue

        if isinstance(result, ProcessedDocument):
            # Archived documents are kept by default (history stays visible),
            # but can be left out with include_archived=False.
            if not include_archived and result.metadata.status.strip().lower() == "archived":
                excluded.append(result.metadata.source_filename)
                continue
            documents.append(result)
        else:
            tables.append(result)

       # Build the registry from the CSVs, now that every file has been read.
    extracted_tables = [t.extracted for t in tables]
    registry = build_registry(extracted_tables)

    # CSV-based relationships (employees, roles, processes, products).
    relationships, unresolved = derive_table_relationships(extracted_tables, registry)

    # Document header relationships, resolved against the registry.
    doc_relationships: list[Relationship] = []
    for doc in documents:
        rels, notes = derive_document_relationships(doc.metadata, registry)
        doc_relationships.extend(rels)
        unresolved.extend(notes)
    relationships.extend(doc_relationships)
        # Step 5: relationships discovered from document text.
    discovered = discover_relationships(
        [(doc.metadata, doc.chunks) for doc in documents], registry
    )
    relationships.extend(discovered)
        # Step 6: location -> checklist and location -> tone guide.
    relationships.extend(derive_location_links([doc.metadata for doc in documents], registry))
        # Validation runs last, over EVERY relationship built above.
    all_metadata = [doc.metadata for doc in documents]
    all_chunks = [chunk for doc in documents for chunk in doc.chunks]
    validation_issues = run_all_validations(
        all_metadata, all_chunks, relationships, registry, skipped
    )
    

    # For now the validator only checks document relationships (their source
    # must be a known document). Step 7 upgrades it to check every relationship.
    #validation_issues = run_all_validations(all_metadata, all_chunks, doc_relationships)

    return PipelineResult(
        documents=documents,
        tables=tables,
        relationships=relationships,
        unresolved_references=unresolved,
        registry=registry,
        skipped=skipped,
        validation_issues=validation_issues,
        excluded=excluded,
    )