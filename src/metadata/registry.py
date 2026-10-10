"""
Entity Registry

A single lookup of every known Nexora entity, built from the CSV tables:
departments, roles, processes, products, locations, employees, documents.

Answers simple questions: does this ID exist? What are its details?
What's the ID for this name? It never invents entities: if it isn't
in a CSV, the registry doesn't know it.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.extraction.extractor import ExtractedTable

# For each CSV (by filename without extension):
#   (entity kind, ID column, name column used for name lookups)
# Employees have no name lookup: KORA links employees by ID, not by name.
_TABLE_SPECS: dict[str, tuple[str, str, str | None]] = {
    "departments": ("departments", "dept_id", "name"),
    "roles": ("roles", "role_id", "title"),
    "processes": ("processes", "process_id", "name"),
    "products": ("products", "product_id", "name"),
    "locations": ("locations", "location_id", "city"),
    "employees": ("employees", "employee_id", None),
    "documents": ("documents", "doc_id", "title"),
}


@dataclass
class EntityRegistry:
    # kind -> entity ID -> list of CSV rows (a list, because one ID can
    # legitimately have several rows, e.g. two versions of a document)
    by_id: dict[str, dict[str, list[dict]]] = field(default_factory=dict)

    # kind -> lowercased name -> entity ID
    by_name: dict[str, dict[str, str]] = field(default_factory=dict)

    # Problems noticed while building (e.g. a row with no ID).
    # Reported, never silently fixed.
    warnings: list[str] = field(default_factory=list)

    def exists(self, kind: str, entity_id: str) -> bool:
        return entity_id in self.by_id.get(kind, {})

    def get(self, kind: str, entity_id: str) -> list[dict]:
        return self.by_id.get(kind, {}).get(entity_id, [])

    def id_for_name(self, kind: str, name: str) -> str | None:
        # Lowercased so "Implementation Manager" and "implementation manager" match.
        return self.by_name.get(kind, {}).get(name.strip().lower())


def build_registry(tables: list[ExtractedTable]) -> EntityRegistry:
    """Build an EntityRegistry from the extracted CSV tables."""
    registry = EntityRegistry()

    for table in tables:
        stem = Path(table.file_info.filename).stem   # "employees.csv" -> "employees"
        spec = _TABLE_SPECS.get(stem)
        if spec is None:
            registry.warnings.append(f"No registry spec for table '{table.file_info.filename}', skipped")
            continue

        kind, id_col, name_col = spec
        ids = registry.by_id.setdefault(kind, {})
        names = registry.by_name.setdefault(kind, {})

        for row_number, row in enumerate(table.rows, start=2):  # row 1 is the header
            entity_id = (row.get(id_col) or "").strip()
            if not entity_id:
                registry.warnings.append(f"{table.file_info.filename} row {row_number}: missing '{id_col}'")
                continue

            ids.setdefault(entity_id, []).append(row)

            if name_col:
                name = (row.get(name_col) or "").strip()
                if name:
                    names[name.lower()] = entity_id
                    
                                # Documents can also be referenced by filename in headers,
            # e.g. "communication_culture.md", so index them by filename too.
            if kind == "documents":
                file_path = (row.get("file_path") or "").strip()
                if file_path:
                    file_names = registry.by_name.setdefault("document_files", {})
                    file_names[Path(file_path).name.lower()] = entity_id

    return registry