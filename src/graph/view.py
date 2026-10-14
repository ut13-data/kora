"""
The graph as ONE user may see it.

    view = GraphView(graph, user, policy, config)       # user=None: everything (backend tools)
    view.neighbors("process:PROC-IMPL-001", "performs", IN)
    view.walk(...), view.path(...), view.summary()      # Phase 3 traversals, already filtered

Rules (same principle as Phase 4 M7: a hidden document is never searched,
quoted, cited or NAMED -- here also never linked or counted):

1. Documents: visible if the access rules let the user read them.
2. Processes and policies: visible if at least one document describing them
   is visible. PROC-FIN-003 is in processes.csv, but its only document is
   restricted, so for an Engineering employee the process does not exist.
3. Topics, terms, checklists, customers, audiences: visible if they have no
   source documents or at least one visible one (a topic also needs its anchor).
4. People, roles, departments, locations, products: visible (the directory).
   A colleague's private details are not: only the columns in the config's
   person_visibility.colleague_fields are shown, and only the links that come
   from those columns (role, department, location, manager). The person
   themself and admins see everything about that person.
5. Links: visible only if both ends are visible AND some evidence is visible.
   Evidence from a hidden document is removed from the link, so a visible
   link never quotes or cites a hidden document either.
6. Everything is computed on the filtered graph, so counts, paths and the
   summary can't reveal hidden things.
"""

from dataclasses import replace

from src.intelligence.graph import BOTH, IN, OUT, Hop, KnowledgeGraph
from src.intelligence.models import Entity
from src.intelligence.relationships import KnowledgeRelationship
from src.retrieval.access import AccessPolicy, User

DIRECTORY_TYPES = {"employee", "role", "department", "location", "product"}
DOCUMENTED_TYPES = {"process", "policy"}
# employee column -> the graph link that comes from it (Phase 3 relationships.py)
FIELD_LINKS = {"role_id": "has_role", "role_title": "has_role", "dept_id": "works_in",
               "location_id": "located_in", "manager_id": "reports_to",
               "international_relocation_germany": "eligible_for"}


class GraphView(KnowledgeGraph):
    def __init__(self, base: KnowledgeGraph, user: User | None = None, policy: AccessPolicy | None = None,
                 colleague_fields: list[str] | None = None, admins: list[str] | None = None):
        # Shares the base graph's data; filtering happens on every read.
        self.base = base
        self.index = base.index
        self.out_edges = base.out_edges
        self.in_edges = base.in_edges
        self.user = user
        self.policy = policy or AccessPolicy()
        self.colleague_fields = set(colleague_fields if colleague_fields is not None else FIELD_LINKS)
        self.admin = user is not None and user.user_id in set(admins or [])
        self._node_ok: dict[str, bool] = {}
        self._rel_cache: dict[str, KnowledgeRelationship | None] = {}
        self._documented_by: dict[str, list[str]] = {}
        for rel in base.relationships:
            if rel.relationship == "documented_by":
                self._documented_by.setdefault(rel.source_key, []).append(rel.target_id)

    @classmethod
    def for_user(cls, base: KnowledgeGraph, user: User | None, config) -> "GraphView":
        """The usual way in: access rules, visible fields and admins from the company config."""
        return cls(base, user, AccessPolicy(config.access_rules), config.colleague_fields, config.admins)

    @property
    def relationships(self) -> list[KnowledgeRelationship]:
        return [r for r in (self.visible_relationship(r) for r in self.base.relationships) if r is not None]

    @relationships.setter
    def relationships(self, value) -> None:      # KnowledgeGraph.__init__ is never called; keep it read-only
        raise AttributeError("GraphView is read-only")

    # ---------------- who sees what ----------------

    def full_access(self, employee_key: str | None = None) -> bool:
        if self.user is None or self.admin:
            return True
        return employee_key is not None and employee_key == f"employee:{self.user.user_id}"

    def can_read(self, document_id: str | None) -> bool:
        return document_id is None or self.policy.can_read(self.user, document_id)

    def visible(self, key: str) -> bool:
        if key not in self._node_ok:
            self._node_ok[key] = self._visible(key)
        return self._node_ok[key]

    def _visible(self, key: str) -> bool:
        e = self.base.node(key)
        if e is None:
            return False
        if self.user is None:
            return True
        if e.entity_type == "document":
            return self.can_read(e.entity_id)
        if e.entity_type in DIRECTORY_TYPES:
            return True
        if e.entity_type in DOCUMENTED_TYPES:
            docs = self._documented_by.get(key) or [d for d in e.source_documents if d == e.entity_id] \
                or e.source_documents
            return not docs or any(self.can_read(d) for d in docs)
        anchor = e.attributes.get("anchor")
        if anchor and anchor != key and not self.visible(anchor):
            return False
        return not e.source_documents or any(self.can_read(d) for d in e.source_documents)

    def _employee_link_ok(self, rel: KnowledgeRelationship) -> bool:
        if rel.source_type != "employee" or self.full_access(rel.source_key):
            return True
        allowed = {FIELD_LINKS[f] for f in self.colleague_fields if f in FIELD_LINKS}
        return rel.relationship in allowed

    def _evidence_ok(self, ev) -> bool:
        if not self.can_read(ev.source_document):
            return False
        return all(self.can_read(cid.rsplit("_", 2)[0]) for cid in ev.chunk_ids)

    def visible_relationship(self, rel: KnowledgeRelationship) -> KnowledgeRelationship | None:
        """The link as this user may see it (hidden evidence removed), or None."""
        if rel.relationship_id not in self._rel_cache:
            out = None
            if self.visible(rel.source_key) and self.visible(rel.target_key) and self._employee_link_ok(rel):
                evidence = [e for e in rel.evidence if self._evidence_ok(e)]
                if evidence:
                    out = rel if len(evidence) == len(rel.evidence) else replace(rel, evidence=evidence)
            self._rel_cache[rel.relationship_id] = out
        return self._rel_cache[rel.relationship_id]

    # ---------------- KnowledgeGraph reads, filtered ----------------

    def node(self, key: str) -> Entity | None:
        if not self.visible(key):
            return None
        e = self.base.node(key)
        if self.user is None:
            return e
        docs = [d for d in e.source_documents if self.can_read(d)]
        attributes = e.attributes
        if e.entity_type == "employee" and not self.full_access(key):
            attributes = {k: v for k, v in e.attributes.items() if k in self.colleague_fields}
        elif e.entity_type == "topic" and len(docs) < len(e.source_documents):
            # keywords were computed from every document, hidden ones included
            attributes = {k: v for k, v in e.attributes.items() if k != "keywords"}
        if docs == e.source_documents and attributes is e.attributes:
            return e
        return replace(e, source_documents=docs, attributes=attributes)

    def hops(self, key: str, relationship: str | None = None, direction: str = OUT,
             current_only: bool = False, min_confidence: float = 0.0) -> list[Hop]:
        if not self.visible(key):
            return []
        found: list[Hop] = []
        if direction in (OUT, BOTH):
            found += [Hop(r, OUT, r.target_key) for r in self.out_edges.get(key, [])]
        if direction in (IN, BOTH):
            found += [Hop(r, IN, r.source_key) for r in self.in_edges.get(key, [])]
        out = []
        for h in found:
            rel = self.visible_relationship(h.relationship)
            if rel is None or (relationship is not None and rel.relationship != relationship):
                continue
            if (current_only and rel.historical) or rel.confidence < min_confidence:
                continue
            out.append(Hop(rel, h.direction, h.node))
        return out

    def summary(self) -> dict:
        visible = {k: self.base.index.entities[k] for k in self.base.index.entities if self.visible(k)}
        s = KnowledgeGraph.summary(self)
        s["nodes"] = len(visible)
        s["isolated_nodes"] = [k for k in s["isolated_nodes"] if k in visible]
        return s

    def entities(self, entity_type: str | None = None) -> list[Entity]:
        """Visible entities (filtered copies), optionally of one type, sorted by name."""
        keys = [k for k, e in self.base.index.entities.items()
                if (entity_type is None or e.entity_type == entity_type) and self.visible(k)]
        return sorted((self.node(k) for k in keys), key=lambda e: (e.entity_type, e.name.lower()))
