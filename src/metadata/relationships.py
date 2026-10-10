"""
Stage 6: Relationships

Builds a single list of relationships from two sources:
  1. Document headers (Related Processes / Related Roles), resolved to
     real registry IDs, or to audiences (groups of people).
  2. CSV tables (employees, roles, processes, products).

Every relationship records its evidence (where it came from).
Values that can't be resolved are REPORTED, never guessed.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.extraction.extractor import ExtractedTable
from src.metadata.document_metadata import DocumentMetadata
from src.metadata.registry import EntityRegistry


class RelationType(Enum):
    # From document headers
    DOCUMENT_RELATED_TO_PROCESS = "document_related_to_process"
    DOCUMENT_RELATED_TO_DOCUMENT = "document_related_to_document"
    DOCUMENT_RELATED_TO_ROLE = "document_related_to_role"
    DOCUMENT_FOR_AUDIENCE = "document_for_audience"

    # From CSV tables
    EMPLOYEE_HAS_ROLE = "employee_has_role"
    EMPLOYEE_AT_LOCATION = "employee_at_location"
    EMPLOYEE_REPORTS_TO = "employee_reports_to"
    ROLE_IN_DEPARTMENT = "role_in_department"
    ROLE_RELATED_TO_PROCESS = "role_related_to_process"
    PROCESS_OWNED_BY_DEPARTMENT = "process_owned_by_department"
    PROCESS_DOCUMENTED_IN = "process_documented_in"
    PRODUCT_OWNED_BY_DEPARTMENT = "product_owned_by_department"
    PRODUCT_DOCUMENTED_IN = "product_documented_in"
        # Discovered from document text (Step 5)
    DOCUMENT_MENTIONS_PROCESS = "document_mentions_process"
    DOCUMENT_MENTIONS_DOCUMENT = "document_mentions_document"
    DOCUMENT_MENTIONS_PRODUCT = "document_mentions_product"
    DOCUMENT_MENTIONS_ROLE = "document_mentions_role"
    DOCUMENT_MENTIONS_LOCATION = "document_mentions_location"
    DOCUMENT_HAS_TOPIC = "document_has_topic"
  
    PROCESS_RELATED_TO_PRODUCT = "process_related_to_product"
      # Location links (Step 6)
    LOCATION_HAS_CHECKLIST = "location_has_checklist"
    CULTURE_TONE_GUIDANCE = "culture_tone_guidance"


@dataclass
class Relationship:
    source: str
    relation_type: RelationType
    target: str
    evidence: str   # where this came from, e.g. "employees.csv: role_id"


# Header values that mean "not specific", so they can't link to anything.
_PLACEHOLDERS = {"multiple", "various", "n/a", "none", "tbd", "all"}

# Role synonyms documented in the Nexora knowledge base itself.
# Only add an entry when a Nexora document says they mean the same thing.
_ROLE_ALIASES = {
    "project lead": "Implementation Manager",           # glossary.md TN-006
    "sdr": "Sales Development Representative",          # glossary.md
    "ae": "Account Executive",                          # glossary.md
    "csm": "Customer Success Manager",                  # glossary.md
    "demand gen manager": "Demand Generation Manager",  # marketing.md roles
}

# Group words that mark a value as an AUDIENCE (a group of people),
# rather than a single role. Role titles are singular; audiences are groups.
_AUDIENCE_KEYWORDS = ("employees", "managers", "new hires", "colleagues")


def _core_value(raw: str) -> str:
    """Drop the explanation in brackets:
    'PROC-SALES-002 (Proposal & Quotation)' -> 'PROC-SALES-002'
    """
    return raw.split("(")[0].strip()


def _lookup_role(name: str, registry: EntityRegistry) -> str | None:
    """Find a role ID by exact title, then by documented alias."""
    role_id = registry.id_for_name("roles", name)
    if role_id is None and name.lower() in _ROLE_ALIASES:
        role_id = registry.id_for_name("roles", _ROLE_ALIASES[name.lower()])
    return role_id


def _resolve_roles(raw: str, registry: EntityRegistry) -> tuple[list[str], list[str]]:
    """Resolve a role value to role IDs.

    1. Try the WHOLE name first, so titles containing "/" survive
       (e.g. "AP/AR Specialist" is ONE role).
    2. Only if that fails, split on "/" for joint owners
       (e.g. "Support Team Lead / Engineering Manager" -> two roles).
    3. If nothing resolves, report the original value once (not in pieces).

    Returns (resolved_role_ids, names_not_found).
    """
    core = _core_value(raw)

    whole = _lookup_role(core, registry)
    if whole:
        return [whole], []

    found: list[str] = []
    missing: list[str] = []
    for part in core.split("/"):
        name = part.strip()
        if not name:
            continue
        role_id = _lookup_role(name, registry)
        if role_id:
            found.append(role_id)
        else:
            missing.append(name)

    if not found:
        return [], [core]   # nothing matched: report it as one value
    return found, missing


def _is_audience(raw: str) -> bool:
    lowered = raw.lower()
    return any(keyword in lowered for keyword in _AUDIENCE_KEYWORDS)


def _audience_id(raw: str) -> str:
    """Build a readable, stable audience ID from the FULL original text.

    'International New Hires (Germany)' -> 'AUD-INTERNATIONAL-NEW-HIRES-GERMANY'
    Qualifiers in brackets are kept, since "(Germany)" changes who it's for.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return f"AUD-{slug.upper()}"


# ---------------- 1. Document header relationships ----------------

def derive_document_relationships(
    metadata: DocumentMetadata, registry: EntityRegistry
) -> tuple[list[Relationship], list[str]]:
    """Resolve a document's Related Processes / Related Roles to real IDs
    or audiences. Returns (relationships, unresolved_notes).
    """
    relationships: list[Relationship] = []
    unresolved: list[str] = []

    if metadata.document_id == "unknown":
        return relationships, unresolved

    # The filename goes into every evidence string, so links from two
    # files sharing one ID (e.g. v1.0 and v2.0) can still be told apart.
    where = metadata.source_filename

    # Related Processes can point at a process ID, a document ID,
    # or a document filename. Try each, in that order.
    for raw in metadata.related_processes:
        core = _core_value(raw)
        if core.lower() in _PLACEHOLDERS:
            unresolved.append(f"{where}: placeholder '{raw}' in Related Processes, skipped")
        elif registry.exists("processes", core):
            relationships.append(Relationship(metadata.document_id,
                RelationType.DOCUMENT_RELATED_TO_PROCESS, core,
                f"{where}: Related Processes"))
        elif registry.exists("documents", core):
            relationships.append(Relationship(metadata.document_id,
                RelationType.DOCUMENT_RELATED_TO_DOCUMENT, core,
                f"{where}: Related Processes"))
        elif (doc_id := registry.id_for_name("document_files", core)) is not None:
            relationships.append(Relationship(metadata.document_id,
                RelationType.DOCUMENT_RELATED_TO_DOCUMENT, doc_id,
                f"{where}: Related Processes (by filename)"))
        else:
            unresolved.append(f"{where}: Related Processes value '{raw}' not found in registry")

    # Related Roles, in order: placeholder -> role -> audience -> unresolved.
    for raw in metadata.related_roles:
        if _core_value(raw).lower() in _PLACEHOLDERS:
            unresolved.append(f"{where}: placeholder '{raw}' in Related Roles, skipped")
            continue

        role_ids, missing = _resolve_roles(raw, registry)
        if role_ids:
            for role_id in role_ids:
                relationships.append(Relationship(metadata.document_id,
                    RelationType.DOCUMENT_RELATED_TO_ROLE, role_id,
                    f"{where}: Related Roles"))
            for name in missing:
                unresolved.append(f"{where}: Related Roles value '{name}' not found in registry")
        elif _is_audience(raw):
            # Original wording kept in the evidence, for traceability.
            relationships.append(Relationship(metadata.document_id,
                RelationType.DOCUMENT_FOR_AUDIENCE, _audience_id(raw),
                f"{where}: Related Roles '{raw}'"))
        else:
            unresolved.append(f"{where}: Related Roles value '{raw}' not found in registry")

    return relationships, unresolved


# ---------------- 2. CSV table relationships ----------------

def _rows(tables: list[ExtractedTable], stem: str) -> list[dict]:
    """Return the rows of the table whose filename is <stem>.csv."""
    for table in tables:
        if Path(table.file_info.filename).stem == stem:
            return table.rows
    return []


def _value(row: dict, column: str) -> str:
    return (row.get(column) or "").strip()


def derive_table_relationships(
    tables: list[ExtractedTable], registry: EntityRegistry
) -> tuple[list[Relationship], list[str]]:
    """Build relationships from the CSV tables. Returns (relationships, unresolved_notes)."""
    relationships: list[Relationship] = []
    unresolved: list[str] = []

    def add(source: str, rtype: RelationType, target: str, evidence: str) -> None:
        # Skip empty cells (e.g. an employee with no manager_id).
        if source and target:
            relationships.append(Relationship(source, rtype, target, evidence))

    for row in _rows(tables, "employees"):
        emp = _value(row, "employee_id")
        add(emp, RelationType.EMPLOYEE_HAS_ROLE, _value(row, "role_id"), "employees.csv: role_id")
        add(emp, RelationType.EMPLOYEE_AT_LOCATION, _value(row, "location_id"), "employees.csv: location_id")
        add(emp, RelationType.EMPLOYEE_REPORTS_TO, _value(row, "manager_id"), "employees.csv: manager_id")

    for row in _rows(tables, "roles"):
        add(_value(row, "role_id"), RelationType.ROLE_IN_DEPARTMENT, _value(row, "dept_id"), "roles.csv: dept_id")

    for row in _rows(tables, "processes"):
        proc = _value(row, "process_id")
        add(proc, RelationType.PROCESS_OWNED_BY_DEPARTMENT, _value(row, "owning_dept_id"), "processes.csv: owning_dept_id")
        add(proc, RelationType.PROCESS_DOCUMENTED_IN, _value(row, "doc_id"), "processes.csv: doc_id")

        # owner_role is a role TITLE (sometimes "A / B" joint owners),
        # so resolve it to role IDs first.
        owner = _value(row, "owner_role")
        if owner:
            role_ids, missing = _resolve_roles(owner, registry)
            for role_id in role_ids:
                add(role_id, RelationType.ROLE_RELATED_TO_PROCESS, proc, "processes.csv: owner_role")
            for name in missing:
                unresolved.append(f"processes.csv {proc}: owner_role '{name}' not found in roles")

    for row in _rows(tables, "products"):
        prod = _value(row, "product_id")
        add(prod, RelationType.PRODUCT_OWNED_BY_DEPARTMENT, _value(row, "owning_dept_id"), "products.csv: owning_dept_id")
        add(prod, RelationType.PRODUCT_DOCUMENTED_IN, _value(row, "doc_id"), "products.csv: doc_id")

    return relationships, unresolved