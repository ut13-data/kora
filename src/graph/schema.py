"""
The graph's schema: the relationship shapes KORA knows, and what each
provenance tier means.

A shape is (source entity type, relationship, target entity type). The list
below is every shape Phase 3 produces for Nexora, written down so that a new
or misspelled one (e.g. a future 'depends_on' or 'aproves') is REPORTED by the
quality check instead of silently flowing into answers. Adding a shape is a
one-line, reviewed change here.
"""

VALID_SHAPES: frozenset[tuple[str, str, str]] = frozenset({
    ("checklist", "documented_by", "document"),
    ("department", "head_located_in", "location"),
    ("department", "headed_by", "role"),
    ("department", "present_in", "location"),
    ("document", "belongs_to", "department"),
    ("document", "discusses", "topic"),
    ("document", "for_audience", "audience"),
    ("document", "mentions", "customer"),
    ("document", "mentions", "document"),
    ("document", "mentions", "location"),
    ("document", "mentions", "process"),
    ("document", "mentions", "product"),
    ("document", "mentions", "role"),
    ("document", "references", "document"),
    ("document", "related_to", "process"),
    ("document", "relevant_to", "role"),
    ("document", "uses_term", "terminology"),
    ("employee", "eligible_for", "checklist"),
    ("employee", "has_role", "role"),
    ("employee", "located_in", "location"),
    ("employee", "reports_to", "employee"),
    ("employee", "works_in", "department"),
    ("location", "has_checklist", "checklist"),
    ("location", "has_tone_guidance", "document"),
    ("policy", "documented_by", "document"),
    ("process", "documented_by", "document"),
    ("process", "governed_by", "policy"),
    ("process", "owned_by", "department"),
    ("process", "related_to", "process"),
    ("process", "related_to", "product"),
    ("process", "triggers", "process"),
    ("process", "uses", "product"),
    ("product", "documented_by", "document"),
    ("product", "owned_by", "department"),
    ("product", "used_by", "customer"),
    ("role", "approves", "process"),
    ("role", "handles_escalations_for", "process"),
    ("role", "in_department", "department"),
    ("role", "owns", "document"),
    ("role", "owns", "process"),
    ("role", "participates_in", "process"),
    ("role", "performs", "process"),
    ("role", "reports_to", "role"),
    ("terminology", "defined_in", "document"),
    ("terminology", "disambiguates", "process"),
    ("topic", "about", "department"),
    ("topic", "about", "document"),
    ("topic", "about", "process"),
    ("topic", "about", "product"),
})

# ---------------- provenance tiers ----------------
# How KORA knows a link, strongest first. Only the first three are DOCUMENTED:
# an answer may state them as facts, each with a citation that is checked
# again at answer time. The last two can only be shown as "possibly related".
QUOTE = "quote"        # a sentence, found word-for-word in a chunk (or the raw file)
HEADER = "header"      # a document header line ("**Owner/Role:** Finance Manager / ...")
RECORD = "record"      # a value in a company table row (processes.csv, row PROC-FIN-002)
MENTION = "mention"    # both things appear in the same text, nothing more
DERIVED = "derived"    # KORA's own analysis: similarity, topic anchoring, computed rules

TIERS = (QUOTE, HEADER, RECORD, MENTION, DERIVED)
DOCUMENTED = frozenset({QUOTE, HEADER, RECORD})

TIER_LABELS = {
    QUOTE: "Documented (quoted sentence)",
    HEADER: "Documented (document header)",
    RECORD: "Documented (company record)",
    MENTION: "Mentioned together (not a stated relationship)",
    DERIVED: "Derived by KORA (not stated anywhere)",
}


def stronger(a: str | None, b: str | None) -> str | None:
    """The stronger of two tiers (None = no verified evidence)."""
    if a is None:
        return b
    if b is None:
        return a
    return a if TIERS.index(a) <= TIERS.index(b) else b
