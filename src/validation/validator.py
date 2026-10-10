"""
Stage 7: Validation

Checks the output of the whole pipeline for structural problems:
missing IDs, duplicates, invalid versions, empty or duplicate chunks,
broken relationships, header vs master-index mismatches,
employees with no manager, and skipped (unsupported) files.

Reports problems. Does NOT fix anything automatically.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from src.chunking.chunker import Chunk
from src.metadata.document_metadata import DocumentMetadata
from src.metadata.registry import EntityRegistry
from src.metadata.relationships import Relationship, RelationType

UNKNOWN = "unknown"


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    severity: Severity
    message: str
    subject: str  # what this issue is about: a filename, ID, or evidence string


# ---------------- Documents ----------------

def validate_metadata_batch(all_metadata: list[DocumentMetadata]) -> list[ValidationIssue]:
    """Check documents for missing IDs, missing titles, invalid versions,
    and duplicate / multi-version Document IDs."""
    issues: list[ValidationIssue] = []

    # ---- Per-document checks ----
    for metadata in all_metadata:
        if metadata.document_id == UNKNOWN:
            issues.append(ValidationIssue(Severity.ERROR,
                "Document is missing a Document ID", metadata.source_filename))

        if metadata.title == UNKNOWN:
            issues.append(ValidationIssue(Severity.WARNING,
                "Document is missing a title", metadata.source_filename))

        # A missing version is "unknown", not "invalid", so only check found ones.
        if metadata.version != UNKNOWN and not _looks_like_valid_version(metadata.version):
            issues.append(ValidationIssue(Severity.WARNING,
                f"Version '{metadata.version}' doesn't look like a standard version format",
                metadata.source_filename))

    # ---- Batch check: same ID used by several documents ----
    by_id: dict[str, list[DocumentMetadata]] = {}
    for m in all_metadata:
        if m.document_id != UNKNOWN:
            by_id.setdefault(m.document_id, []).append(m)

    for doc_id, docs in by_id.items():
        if len(docs) < 2:
            continue
        versions = [d.version for d in docs]
        if len(set(versions)) < len(versions):
            # Same ID AND same version: a genuine duplicate.
            issues.append(ValidationIssue(Severity.ERROR,
                f"Document ID '{doc_id}' has duplicate entries with the same version", doc_id))
        else:
            # Same ID, different versions: a version history, kept on purpose.
            active = [d for d in docs if d.status.lower() == "active"]
            issues.append(ValidationIssue(Severity.WARNING,
                f"Document ID '{doc_id}' has {len(docs)} versions "
                f"({', '.join(versions)}); {len(active)} marked Active", doc_id))

    return issues


def _looks_like_valid_version(version: str) -> bool:
    """Very simple check: expects something like '1.0' or '2.3.1'."""
    parts = version.split(".")
    return len(parts) >= 2 and all(part.isdigit() for part in parts)


def _same_value(a: str, b: str) -> bool:
    """Compare a header value with a CSV value.
    '2.0' and '2' count as equal (numbers compared as numbers);
    'Active' and 'active' count as equal (text compared case-insensitively)."""
    try:
        return float(a) == float(b)
    except ValueError:
        return a.strip().lower() == b.strip().lower()


def validate_against_document_index(
    all_metadata: list[DocumentMetadata], registry: EntityRegistry
) -> list[ValidationIssue]:
    """Compare each file's header with its row in documents.csv (the master index)."""
    issues: list[ValidationIssue] = []
    file_index = registry.by_name.get("document_files", {})

    for m in all_metadata:
        csv_id = file_index.get(m.source_filename.lower())

        if csv_id is None:
            issues.append(ValidationIssue(Severity.WARNING,
                "File is not listed in documents.csv (the master index)", m.source_filename))
            continue

        if m.document_id != csv_id:
            # Identity disagrees between two sources: serious, so ERROR.
            issues.append(ValidationIssue(Severity.ERROR,
                f"Header ID '{m.document_id}' doesn't match documents.csv ID '{csv_id}'",
                m.source_filename))
            continue  # rows can't be compared reliably if the IDs differ

        row = registry.get("documents", csv_id)[0]
        for field_name in ("version", "status"):
            header_value = getattr(m, field_name)
            csv_value = (row.get(field_name) or "").strip()
            if header_value != UNKNOWN and csv_value and not _same_value(header_value, csv_value):
                issues.append(ValidationIssue(Severity.WARNING,
                    f"{field_name.capitalize()} mismatch: header says '{header_value}', "
                    f"documents.csv says '{csv_value}'", m.source_filename))

    return issues


# ---------------- Chunks ----------------

def validate_chunks_batch(all_chunks: list[Chunk]) -> list[ValidationIssue]:
    """Check chunks for empty text, duplicate chunk IDs,
    and duplicate text within the same document."""
    issues: list[ValidationIssue] = []

    for chunk in all_chunks:
        if not chunk.chunk_text.strip():
            issues.append(ValidationIssue(Severity.ERROR, "Chunk has empty text", chunk.chunk_id))

    # Chunk IDs are stable now, so two chunks sharing one is a real error.
    for chunk_id, count in Counter(c.chunk_id for c in all_chunks).items():
        if count > 1:
            issues.append(ValidationIssue(Severity.ERROR,
                f"Chunk ID is used {count} times", chunk_id))

    # Duplicate text within the same document (across different documents
    # is normal, e.g. shared boilerplate).
        # Duplicate text within the same document VERSION (identical text across
    # two versions of one document is normal, e.g. an unchanged "Purpose").
    by_document: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in all_chunks:
        by_document.setdefault((chunk.document_id, chunk.version), []).append(chunk)

    for (document_id, version), chunks in by_document.items():
        for text, count in Counter(c.chunk_text for c in chunks).items():
            if count > 1:
                issues.append(ValidationIssue(Severity.WARNING,
                    f"Duplicate chunk text appears {count} times in this document",
                    f"{document_id} v{version}"))
    return issues


# ---------------- Relationships ----------------

# What kind of entity sits at each end of each relationship type.
_ENDPOINTS: dict[RelationType, tuple[str, str]] = {
    RelationType.DOCUMENT_RELATED_TO_PROCESS: ("document", "process"),
    RelationType.DOCUMENT_RELATED_TO_DOCUMENT: ("document", "document"),
    RelationType.DOCUMENT_RELATED_TO_ROLE: ("document", "role"),
    RelationType.DOCUMENT_FOR_AUDIENCE: ("document", "audience"),
    RelationType.EMPLOYEE_HAS_ROLE: ("employee", "role"),
    RelationType.EMPLOYEE_AT_LOCATION: ("employee", "location"),
    RelationType.EMPLOYEE_REPORTS_TO: ("employee", "employee"),
    RelationType.ROLE_IN_DEPARTMENT: ("role", "department"),
    RelationType.ROLE_RELATED_TO_PROCESS: ("role", "process"),
    RelationType.PROCESS_OWNED_BY_DEPARTMENT: ("process", "department"),
    RelationType.PROCESS_DOCUMENTED_IN: ("process", "document"),
    RelationType.PRODUCT_OWNED_BY_DEPARTMENT: ("product", "department"),
    RelationType.PRODUCT_DOCUMENTED_IN: ("product", "document"),
    RelationType.DOCUMENT_MENTIONS_PROCESS: ("document", "process"),
    RelationType.DOCUMENT_MENTIONS_DOCUMENT: ("document", "document"),
    RelationType.DOCUMENT_MENTIONS_PRODUCT: ("document", "product"),
    RelationType.DOCUMENT_MENTIONS_ROLE: ("document", "role"),
    RelationType.DOCUMENT_MENTIONS_LOCATION: ("document", "location"),
    RelationType.DOCUMENT_HAS_TOPIC: ("document", "topic"),
    RelationType.PROCESS_RELATED_TO_PRODUCT: ("process", "product"),
    RelationType.LOCATION_HAS_CHECKLIST: ("location", "document"),
    RelationType.CULTURE_TONE_GUIDANCE: ("location", "document"),
}

# Entity kind -> registry table name.
_KIND_TO_TABLE = {
    "process": "processes",
    "role": "roles",
    "department": "departments",
    "location": "locations",
    "employee": "employees",
    "product": "products",
}

# Kinds that KORA creates itself, so there's no table to check against;
# only their ID format is checked.
_SELF_DEFINED_PREFIX = {"audience": "AUD-", "topic": "TOPIC-"}


def _entity_exists(kind: str, entity_id: str, registry: EntityRegistry,
                   known_doc_ids: set[str]) -> bool:
    if kind == "document":
        # A document is known if a file header OR documents.csv has its ID.
        return entity_id in known_doc_ids or registry.exists("documents", entity_id)
    if kind in _SELF_DEFINED_PREFIX:
        return entity_id.startswith(_SELF_DEFINED_PREFIX[kind])
    return registry.exists(_KIND_TO_TABLE[kind], entity_id)


def validate_relationships_batch(
    all_relationships: list[Relationship],
    all_metadata: list[DocumentMetadata],
    registry: EntityRegistry,
) -> list[ValidationIssue]:
    """Check that both ends of every relationship exist."""
    issues: list[ValidationIssue] = []
    known_doc_ids = {m.document_id for m in all_metadata if m.document_id != UNKNOWN}

    for rel in all_relationships:
        endpoints = _ENDPOINTS.get(rel.relation_type)
        if endpoints is None:
            # A new relation type was added without telling the validator.
            issues.append(ValidationIssue(Severity.WARNING,
                f"No validation rule for relation type '{rel.relation_type.value}'", rel.evidence))
            continue

        source_kind, target_kind = endpoints
        if not _entity_exists(source_kind, rel.source, registry, known_doc_ids):
            issues.append(ValidationIssue(Severity.ERROR,
                f"{rel.relation_type.value}: source {source_kind} '{rel.source}' doesn't exist",
                rel.evidence))
        if not _entity_exists(target_kind, rel.target, registry, known_doc_ids):
            issues.append(ValidationIssue(Severity.ERROR,
                f"{rel.relation_type.value}: target {target_kind} '{rel.target}' doesn't exist",
                rel.evidence))

    return issues


# ---------------- Employees ----------------

def validate_employees(registry: EntityRegistry) -> list[ValidationIssue]:
    """One summary warning for employees with no manager_id.
    Normally only the top of the org (e.g. the CEO) should have none."""
    missing = sorted(
        emp_id for emp_id, rows in registry.by_id.get("employees", {}).items()
        if not (rows[0].get("manager_id") or "").strip()
    )
    if not missing:
        return []
    return [ValidationIssue(Severity.WARNING,
        f"{len(missing)} employee(s) have no manager_id: {', '.join(missing)}",
        "employees.csv")]


# ---------------- Everything ----------------

def run_all_validations(
    all_metadata: list[DocumentMetadata],
    all_chunks: list[Chunk],
    all_relationships: list[Relationship],
    registry: EntityRegistry,
    skipped: list[str],
) -> list[ValidationIssue]:
    """Run every validation check and return the combined list of issues."""
    issues: list[ValidationIssue] = []

    # Files the pipeline couldn't process (e.g. unsupported type).
    for note in skipped:
        issues.append(ValidationIssue(Severity.WARNING, f"File skipped: {note}", note.split(":")[0]))

    issues.extend(validate_metadata_batch(all_metadata))
    issues.extend(validate_against_document_index(all_metadata, registry))
    issues.extend(validate_chunks_batch(all_chunks))
    issues.extend(validate_relationships_batch(all_relationships, all_metadata, registry))
    issues.extend(validate_employees(registry))
    return issues