"""
Stage 6: Metadata Layer (Relationships)

Defines a generic structure for recording relationships between entities
(documents, roles, processes, departments, etc.), and derives the
relationships we can honestly support today from data already extracted
in Stage 4.

This is a FOUNDATION for future entity extraction and relationship
discovery -- not a full relationship-extraction engine. We only record
what we can trace directly back to real extracted data.
"""

from dataclasses import dataclass
from enum import Enum

from src.metadata.document_metadata import DocumentMetadata


class RelationType(Enum):
    """The relationship kinds KORA knows about, from the brief's list.

    Not all of these can be POPULATED yet (e.g. Employee -> Location needs
    entity extraction we haven't built). Defining them now gives later
    phases a fixed vocabulary to extend, rather than inventing ad hoc
    strings as they go.
    """
    EMPLOYEE_HAS_ROLE = "employee_has_role"
    ROLE_IN_DEPARTMENT = "role_in_department"
    ROLE_RELATED_TO_PROCESS = "role_related_to_process"
    PROCESS_RELATED_TO_PRODUCT = "process_related_to_product"
    DOCUMENT_RELATED_TO_PROCESS = "document_related_to_process"
    DOCUMENT_RELATED_TO_ROLE = "document_related_to_role"
    EMPLOYEE_AT_LOCATION = "employee_at_location"
    LOCATION_HAS_CHECKLIST = "location_has_checklist"
    CULTURE_TONE_GUIDANCE = "culture_tone_guidance"


@dataclass
class Relationship:
    source: str
    relation_type: RelationType
    target: str


def derive_document_relationships(metadata: DocumentMetadata) -> list[Relationship]:
    """Build Relationships that we can honestly derive from a document's
    already-extracted metadata.

    We only create a relationship when both sides are real, known values
    (never "unknown") -- an unknown document_id can't meaningfully relate
    to anything.
    """
    relationships: list[Relationship] = []

    if metadata.document_id == "unknown":
        return relationships

    for process in metadata.related_processes:
        relationships.append(
            Relationship(
                source=metadata.document_id,
                relation_type=RelationType.DOCUMENT_RELATED_TO_PROCESS,
                target=process,
            )
        )

    for role in metadata.related_roles:
        relationships.append(
            Relationship(
                source=metadata.document_id,
                relation_type=RelationType.DOCUMENT_RELATED_TO_ROLE,
                target=role,
            )
        )

    return relationships