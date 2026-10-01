"""
Stage 7: Validation

Checks the output of the whole pipeline (metadata, chunks, relationships)
for structural problems: missing IDs, duplicates, invalid versions,
unsupported types, empty documents, broken references.

Reports problems. Does NOT fix anything automatically.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from src.chunking.chunker import Chunk
from src.metadata.document_metadata import DocumentMetadata
from src.metadata.relationships import Relationship

UNKNOWN = "unknown"


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    severity: Severity
    message: str
    subject: str  # the document_id, chunk_id, etc. this issue is about


def validate_metadata_batch(all_metadata: list[DocumentMetadata]) -> list[ValidationIssue]:
    """Check a batch of DocumentMetadata for missing IDs, missing titles,
    invalid versions, and duplicate / multi-version Document IDs.

    Reports problems only. Nothing is fixed or removed automatically.
    """
    issues: list[ValidationIssue] = []

    # ---- Per-document checks: each document can be judged on its own ----
    for metadata in all_metadata:

        # A missing ID is serious: chunks and relationships hang off this ID.
        # Subject is the filename, since there's no ID to point at.
        if metadata.document_id == UNKNOWN:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                message="Document is missing a Document ID",
                subject=metadata.source_filename,
            ))

        # A missing title is less serious (the document is still usable),
        # so this is only a warning.
        if metadata.title == UNKNOWN:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                message="Document is missing a title",
                subject=metadata.source_filename,
            ))

        # Only check the format if a version was actually found.
        # A missing version is "unknown", not "invalid".
        if metadata.version != UNKNOWN and not _looks_like_valid_version(metadata.version):
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                message=f"Version '{metadata.version}' doesn't look like a standard version format",
                subject=metadata.source_filename,
            ))

    # ---- Batch-level checks: these need every document at once ----

    # Group documents by their ID, e.g.
    #   {"PROC-IMPL-001": [v2.0 metadata, v1.0 metadata], "DOC-FAQ-001": [...]}
    # Documents with an unknown ID are left out, since "unknown" isn't a real ID
    # (they were already reported above).
    by_id: dict[str, list[DocumentMetadata]] = {}
    for m in all_metadata:
        if m.document_id != UNKNOWN:
            by_id.setdefault(m.document_id, []).append(m)

    for doc_id, docs in by_id.items():
        # One document per ID is the normal, healthy case.
        if len(docs) < 2:
            continue

        versions = [d.version for d in docs]

        # A set drops repeats. If the set is shorter than the list,
        # at least two documents share BOTH the ID and the version:
        # that's a genuine duplicate, not a version history.
        if len(set(versions)) < len(versions):
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                message=f"Document ID '{doc_id}' has duplicate entries with the same version",
                subject=doc_id,
            ))
        else:
            # Same ID, different versions: a legitimate version history
            # (e.g. an archived v1.0 alongside an active v2.0). We keep both
            # and report it as a warning, including how many are Active.
            # More than one Active version WOULD be a real conflict to fix.
            active = [d for d in docs if d.status.lower() == "active"]
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                message=(f"Document ID '{doc_id}' has {len(docs)} versions "
                         f"({', '.join(versions)}); {len(active)} marked Active"),
                subject=doc_id,
            ))

    return issues


def _looks_like_valid_version(version: str) -> bool:
    """Very simple check: expects something like '1.0' or '2.3.1'."""
    parts = version.split(".")
    return len(parts) >= 2 and all(part.isdigit() for part in parts)


def validate_chunks_batch(all_chunks: list[Chunk]) -> list[ValidationIssue]:
    """Check a batch of Chunks for empty text and duplicate chunk content
    within the same document.
    """
    issues: list[ValidationIssue] = []

    for chunk in all_chunks:
        if not chunk.chunk_text.strip():
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                message="Chunk has empty text",
                subject=chunk.chunk_id,
            ))

    # Duplicate chunk text within the same document (checked per-document,
    # since identical boilerplate text in two DIFFERENT documents is normal).
    by_document: dict[str, list[Chunk]] = {}
    for chunk in all_chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    for document_id, chunks in by_document.items():
        text_counts = Counter(c.chunk_text for c in chunks)
        for text, count in text_counts.items():
            if count > 1:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    message=f"Duplicate chunk text appears {count} times in this document",
                    subject=document_id,
                ))

    return issues


def validate_relationships_batch(
    all_relationships: list[Relationship],
    all_metadata: list[DocumentMetadata],
) -> list[ValidationIssue]:
    """Check that every relationship's source document actually exists
    in the metadata batch -- i.e. no relationship points from a document
    ID we never actually saw.
    """
    issues: list[ValidationIssue] = []
    known_document_ids = {m.document_id for m in all_metadata if m.document_id != UNKNOWN}

    for relationship in all_relationships:
        if relationship.source not in known_document_ids:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                message=f"Relationship references unknown source document '{relationship.source}'",
                subject=relationship.source,
            ))

    return issues


def run_all_validations(
    all_metadata: list[DocumentMetadata],
    all_chunks: list[Chunk],
    all_relationships: list[Relationship],
) -> list[ValidationIssue]:
    """Run every validation check and return the combined list of issues."""
    issues: list[ValidationIssue] = []
    issues.extend(validate_metadata_batch(all_metadata))
    issues.extend(validate_chunks_batch(all_chunks))
    issues.extend(validate_relationships_batch(all_relationships, all_metadata))
    return issues