"""
Phase 3 data model: the shapes every intelligence module produces.

Three rules are built into these classes, not left to convention:
  1. Every record has a stable ID.
  2. Every record keeps its evidence (where it came from).
  3. Confidence is a number from 0 to 1 whose meaning is written down below.
"""

from dataclasses import asdict, dataclass, field

# ---------------- Confidence scale ----------------
# Each level says HOW we know something, not how sure we "feel".
# Anything below CONF_KEYWORD is a suggestion for a human, never a fact.
CONF_STRUCTURED = 1.0    # read directly from a CSV column or a document header field
CONF_DOCUMENTED = 0.95   # a Nexora document explicitly states it (e.g. "Same role.")
CONF_PATTERN = 0.85      # deterministic text pattern inside a known section
CONF_KEYWORD = 0.7       # name or keyword match in free text (Phase 2 discovery)
CONF_SIMILARITY = 0.5    # string or statistical similarity: needs human review

# ---------------- Entity types ----------------
EMPLOYEE = "employee"
DEPARTMENT = "department"
ROLE = "role"
PROCESS = "process"
PRODUCT = "product"
CUSTOMER = "customer"
POLICY = "policy"
DOCUMENT = "document"
LOCATION = "location"
TOPIC = "topic"
CHECKLIST = "checklist"
TERMINOLOGY = "terminology"
AUDIENCE = "audience"   # groups like "All Employees" (Phase 2 decision, kept)

ENTITY_TYPES = {
    EMPLOYEE, DEPARTMENT, ROLE, PROCESS, PRODUCT, CUSTOMER, POLICY,
    DOCUMENT, LOCATION, TOPIC, CHECKLIST, TERMINOLOGY, AUDIENCE,
}


def entity_key(entity_type: str, entity_id: str) -> str:
    """One globally unique key per entity.

    Needed because Nexora reuses IDs across kinds on purpose:
    PROC-IT-001 is a process, a document AND a policy. As keys they are
    'process:PROC-IT-001', 'document:PROC-IT-001', 'policy:PROC-IT-001'.
    """
    return f"{entity_type}:{entity_id}"


@dataclass
class Entity:
    entity_id: str                 # the ID Nexora itself uses, unchanged
    entity_type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)  # document IDs that state or discuss it
    sources: list[str] = field(default_factory=list)           # evidence: "employees.csv: employee_id"
    confidence: float = CONF_STRUCTURED
    attributes: dict = field(default_factory=dict)             # type-specific details

    @property
    def key(self) -> str:
        return entity_key(self.entity_type, self.entity_id)

    def to_dict(self) -> dict:
        return {"key": self.key, **asdict(self)}


@dataclass
class PotentialDuplicate:
    """Two things that MAY be the same. Reported for review, never merged."""
    duplicate_id: str
    entity_a: str          # entity key, e.g. "role:ROLE-022"
    entity_b: str          # entity key, or "unresolved:<raw text>" for a name with no entity
    method: str            # how it was found, e.g. "documented_alias", "name_similarity"
    reason: str            # one plain sentence a person can check
    confidence: float
    evidence: list[str] = field(default_factory=list)
    status: str = "unreviewed"   # KORA never sets this to "merged" on its own

    def to_dict(self) -> dict:
        return asdict(self)
