"""
Step 6: Location links

Links each office location to:
  - the settling-in checklists for its country   (LOCATION_HAS_CHECKLIST)
  - the communication tone guide for its country (CULTURE_TONE_GUIDANCE)

The country comes from locations.csv; the document type and title come
from each document's header. A link is made only when the title names
the country -- never assumed.
"""

import re
from collections import defaultdict

from src.metadata.document_metadata import DocumentMetadata
from src.metadata.registry import EntityRegistry
from src.metadata.relationships import Relationship, RelationType

# Document type (from the header, lowercased) -> relationship it produces.
_TYPE_TO_RELATION = {
    "checklist": RelationType.LOCATION_HAS_CHECKLIST,
    "cultural communication guidance": RelationType.CULTURE_TONE_GUIDANCE,
}

# Other names a country goes by in Nexora's own documents.
# Key: name used in a document title. Value: country as in locations.csv.
_COUNTRY_ALIASES = {
    "united states": "usa",   # us_workplace_tone.md title vs locations.csv
}


def _country_names(country: str) -> list[str]:
    """All the ways a country can appear in a title: 'usa' -> ['usa', 'united states']."""
    return [country] + [alias for alias, target in _COUNTRY_ALIASES.items() if target == country]


def derive_location_links(
    all_metadata: list[DocumentMetadata], registry: EntityRegistry
) -> list[Relationship]:
    # country (lowercase) -> location IDs in that country, from locations.csv
    locations_by_country: dict[str, list[str]] = defaultdict(list)
    for loc_id, rows in registry.by_id.get("locations", {}).items():
        country = (rows[0].get("country") or "").strip().lower()
        if country:
            locations_by_country[country].append(loc_id)

    relationships: list[Relationship] = []

    for metadata in all_metadata:
        if metadata.document_id == "unknown":
            continue

        relation = _TYPE_TO_RELATION.get(metadata.document_type.strip().lower())
        if relation is None:
            continue   # not a checklist or tone guide

        title = metadata.title.lower()
        for country, loc_ids in locations_by_country.items():
            for name in _country_names(country):
                if re.search(rf"\b{re.escape(name)}\b", title):
                    for loc_id in sorted(loc_ids):
                        relationships.append(Relationship(
                            loc_id, relation, metadata.document_id,
                            f"{metadata.source_filename}: title names '{name}', "
                            f"type '{metadata.document_type}'"))
                    break   # one match per country is enough

    return relationships