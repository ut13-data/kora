"""
Step 5: Discovery

Scans document text (chunk by chunk) for mentions of entities KORA
already knows: processes, documents, products, roles, locations,
and topics taken from Nexora's own glossary.

This is keyword matching, NOT understanding. Every discovered link
stores evidence (filename + first chunk) so it can be checked.

Departments are deliberately NOT matched: names like "Sales" or
"Operations" are everyday words and would create false links.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from src.chunking.chunker import Chunk
from src.metadata.document_metadata import DocumentMetadata
from src.metadata.registry import EntityRegistry
from src.metadata.relationships import Relationship, RelationType, _ROLE_ALIASES

# Short product names used in Nexora's own documents.
# Only add an entry when Nexora documents actually use the short form.
_PRODUCT_ALIASES = {
    "NexView": "nexview analytics",   # used throughout nexview_analytics.md
}

# Which relationship each kind of mention produces.
_KIND_TO_RELATION = {
    "process": RelationType.DOCUMENT_MENTIONS_PROCESS,
    "document": RelationType.DOCUMENT_MENTIONS_DOCUMENT,
    "product": RelationType.DOCUMENT_MENTIONS_PRODUCT,
    "role": RelationType.DOCUMENT_MENTIONS_ROLE,
    "location": RelationType.DOCUMENT_MENTIONS_LOCATION,
    "topic": RelationType.DOCUMENT_HAS_TOPIC,
}

# Header lines like "**Related Roles:** ..." -- already handled in Step 3/4,
# so they're ignored here to avoid duplicate links.
_HEADER_LINE = re.compile(r"^\*\*[^*]+:\*\*")

# A two-column Markdown table row: "| Term | Definition |"
_TABLE_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")


@dataclass
class _Matcher:
    pattern: re.Pattern
    kind: str      # "process", "product", "role", ...
    target: str    # the entity ID this pattern points to


def _compile(term: str, case_sensitive: bool) -> re.Pattern:
    """Whole-word pattern: matches 'NexFlow' but not 'NexFlowX',
    and 'PROC-IMPL-001' but not 'PROC-IMPL-001-V1'."""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", flags)


def _is_acronym(term: str) -> bool:
    """'ARR', 'SOW', 'P1' -> True. Acronyms are matched case-sensitively,
    so 'arr' inside normal text doesn't count."""
    return term.isupper()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-").upper()


def _glossary_topics(docs: list[tuple[DocumentMetadata, list[Chunk]]]) -> dict[str, list[str]]:
    """Read topics from the glossary table. Returns {topic_id: [terms]}.

    'P1 / P2 / P3 / P4' becomes ONE topic with four searchable terms.
    """
    topics: dict[str, list[str]] = {}
    for metadata, chunks in docs:
        if metadata.document_type.lower() != "glossary":
            continue
        for chunk in chunks:
            for line in chunk.chunk_text.splitlines():
                match = _TABLE_ROW.match(line.strip())
                if not match:
                    continue
                term = match.group(1)
                # Skip the header row ("Term") and the separator row ("---").
                if term.lower() == "term" or set(term) <= set("-: "):
                    continue
                terms = [t.strip() for t in term.split("/") if t.strip()]
                topics[f"TOPIC-{_slug(term)}"] = terms
    return topics


def _build_matchers(registry: EntityRegistry, topics: dict[str, list[str]]) -> list[_Matcher]:
    """Turn every known entity name/ID into a compiled pattern."""
    matchers: list[_Matcher] = []

    def add(term: str, kind: str, target: str, case_sensitive: bool) -> None:
        if term:
            matchers.append(_Matcher(_compile(term, case_sensitive), kind, target))

    # Processes: by ID (exact) and by name.
    for proc_id in registry.by_id.get("processes", {}):
        add(proc_id, "process", proc_id, True)
    for name, proc_id in registry.by_name.get("processes", {}).items():
        add(name, "process", proc_id, False)

    # Documents: by ID only (titles are long and rarely quoted exactly).
    for doc_id in registry.by_id.get("documents", {}):
        add(doc_id, "document", doc_id, True)

    # Products: by name, plus documented short names.
    product_names = registry.by_name.get("products", {})
    for name, prod_id in product_names.items():
        add(name, "product", prod_id, False)
    for short, full_name in _PRODUCT_ALIASES.items():
        if full_name in product_names:
            add(short, "product", product_names[full_name], True)

    # Roles: by title, plus documented aliases (e.g. "Project Lead").
    role_names = registry.by_name.get("roles", {})
    for title, role_id in role_names.items():
        add(title, "role", role_id, False)
    for alias, full_title in _ROLE_ALIASES.items():
        role_id = role_names.get(full_title.lower())
        if role_id:
            # Acronym aliases (stored lowercase) are matched as uppercase.
            term = alias.upper() if len(alias) <= 3 else alias
            add(term, "role", role_id, len(alias) <= 3)

    # Locations: by city name and by location ID.
    for city, loc_id in registry.by_name.get("locations", {}).items():
        add(city, "location", loc_id, False)
    for loc_id in registry.by_id.get("locations", {}):
        add(loc_id, "location", loc_id, True)

    # Topics from the glossary. Skip terms that are really role acronyms
    # (AE, SDR, CSM), since those are already matched as roles above.
    for topic_id, terms in topics.items():
        for term in terms:
            if term.lower() in _ROLE_ALIASES:
                continue
            add(term, "topic", topic_id, _is_acronym(term))

    return matchers


def _strip_header(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _HEADER_LINE.match(line.strip()))

def _version_key(version: str) -> str:
    """'2.0', '2' and '2.00' all become '2', so a version from a CSV
    and a version from a document header compare as equal."""
    try:
        return f"{float(version):g}"
    except ValueError:
        return version.strip().lower()

def discover_relationships(
    docs: list[tuple[DocumentMetadata, list[Chunk]]], registry: EntityRegistry
) -> list[Relationship]:
    """Find entity mentions in every document's chunks and return them as
    relationships. Also derives Process -> Product: if a document is the
    official document for a process, every product it mentions is linked
    to that process.
    """
    topics = _glossary_topics(docs)
    matchers = _build_matchers(registry, topics)

    # Which process (if any) each document describes, from processes.csv doc_id.
        # (document ID, version) -> process. Version is part of the key because
    # two processes can point at the same document ID (PROC-IMPL-001 v2.0
    # and the archived PROC-IMPL-001-V1 v1.0). Keyed by ID alone, the later
    # row silently overwrote the earlier one.
    doc_to_process: dict[tuple[str, str], str] = {}
    for proc_id, rows in registry.by_id.get("processes", {}).items():
        for row in rows:
            doc_id = (row.get("doc_id") or "").strip()
            if doc_id:
                doc_to_process[(doc_id, _version_key(row.get("version") or ""))] = proc_id
    relationships: list[Relationship] = []

    for metadata, chunks in docs:
        if metadata.document_id == "unknown":
            continue

        # (kind, target) -> chunk IDs where it was mentioned
        hits: dict[tuple[str, str], list[str]] = defaultdict(list)

        for chunk in chunks:
            text = _strip_header(chunk.chunk_text)
            for m in matchers:
                # A document mentioning its own ID isn't a useful link.
                if m.target == metadata.document_id:
                    continue
                if m.pattern.search(text):
                    found_in = hits[(m.kind, m.target)]
                    if chunk.chunk_id not in found_in:
                        found_in.append(chunk.chunk_id)

        for (kind, target), chunk_ids in sorted(hits.items()):
            evidence = (f"{metadata.source_filename}: mentioned in "
                        f"{len(chunk_ids)} chunk(s), first in {chunk_ids[0]}")
            relationships.append(Relationship(
                metadata.document_id, _KIND_TO_RELATION[kind], target, evidence))

                        # Process -> Product, via the process's official document
            # (matched on document ID AND version).
            process_id = doc_to_process.get(
                (metadata.document_id, _version_key(metadata.version)))
            if kind == "product" and process_id:
                relationships.append(Relationship(
                    process_id, RelationType.PROCESS_RELATED_TO_PRODUCT, target, evidence))

    return relationships