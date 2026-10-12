"""
A lightweight knowledge graph: entities as nodes, relationships as edges.

No graph database: two dictionaries (outgoing and incoming edges per node)
are enough for ~400 nodes and a few thousand edges, and anyone can read them.

    graph = KnowledgeGraph(index, relationships)
    graph.neighbors("employee:EMP-0077", "has_role")
    graph.walk("employee:EMP-0077", [("has_role", OUT), ("in_department", OUT), ("owned_by", IN)])
    graph.path("employee:EMP-0077", "product:PROD-NEXFLOW")
    graph.explain(relationship)     # "... because customer_onboarding.md (step 2) states: '...'"

Archived knowledge stays in the graph. Pass current_only=True to leave it out.
"""

from collections import Counter, deque
from dataclasses import dataclass

from src.intelligence.entities import EntityIndex
from src.intelligence.models import Entity
from src.intelligence.relationships import KnowledgeRelationship

OUT, IN, BOTH = "out", "in", "both"


@dataclass
class Hop:
    relationship: KnowledgeRelationship
    direction: str          # OUT: followed source -> target; IN: target -> source
    node: str               # entity key reached by this hop


class KnowledgeGraph:
    def __init__(self, index: EntityIndex, relationships: list[KnowledgeRelationship]):
        self.index = index
        self.relationships = relationships
        self.out_edges: dict[str, list[KnowledgeRelationship]] = {}
        self.in_edges: dict[str, list[KnowledgeRelationship]] = {}
        for rel in relationships:
            self.out_edges.setdefault(rel.source_key, []).append(rel)
            self.in_edges.setdefault(rel.target_key, []).append(rel)

    # ---------- basics ----------

    def node(self, key: str) -> Entity | None:
        return self.index.entities.get(key)

    def label(self, key: str) -> str:
        e = self.node(key)
        return f"{e.name} ({e.entity_id})" if e else key

    def hops(self, key: str, relationship: str | None = None, direction: str = OUT,
             current_only: bool = False, min_confidence: float = 0.0) -> list[Hop]:
        found: list[Hop] = []
        if direction in (OUT, BOTH):
            found += [Hop(r, OUT, r.target_key) for r in self.out_edges.get(key, [])]
        if direction in (IN, BOTH):
            found += [Hop(r, IN, r.source_key) for r in self.in_edges.get(key, [])]
        return [h for h in found
                if (relationship is None or h.relationship.relationship == relationship)
                and (not current_only or not h.relationship.historical)
                and h.relationship.confidence >= min_confidence]

    def neighbors(self, key: str, relationship: str | None = None, direction: str = OUT,
                  current_only: bool = False, min_confidence: float = 0.0) -> list[str]:
        return sorted({h.node for h in self.hops(key, relationship, direction, current_only, min_confidence)})

    # ---------- traversal ----------

    def walk(self, start: str, steps: list[tuple[str, str]], current_only: bool = False,
             min_confidence: float = 0.0) -> list[list[Hop]]:
        """Follow a fixed sequence of (relationship, direction) steps.
        Returns every complete path, so each answer carries its own evidence."""
        paths: list[list[Hop]] = [[]]
        for relationship, direction in steps:
            extended = []
            for path in paths:
                here = path[-1].node if path else start
                for hop in self.hops(here, relationship, direction, current_only, min_confidence):
                    if hop.node != start and all(hop.node != h.node for h in path):
                        extended.append(path + [hop])
            paths = extended
        return paths

    def reach(self, start: str, steps: list[tuple[str, str]], **kw) -> list[str]:
        """Just the end nodes of walk()."""
        return sorted({path[-1].node for path in self.walk(start, steps, **kw) if path})

    def path(self, start: str, end: str, max_depth: int = 4, current_only: bool = False,
             min_confidence: float = 0.0) -> list[Hop] | None:
        """Shortest connection between two entities, in either direction."""
        queue: deque[tuple[str, list[Hop]]] = deque([(start, [])])
        seen = {start}
        while queue:
            key, so_far = queue.popleft()
            if key == end:
                return so_far
            if len(so_far) >= max_depth:
                continue
            for hop in self.hops(key, None, BOTH, current_only, min_confidence):
                if hop.node not in seen:
                    seen.add(hop.node)
                    queue.append((hop.node, so_far + [hop]))
        return None

    # ---------- explanations ----------

    def explain(self, rel: KnowledgeRelationship) -> str:
        """One relationship, in words, with every piece of evidence behind it."""
        lines = [f"{self.label(rel.source_key)} --{rel.relationship}--> {self.label(rel.target_key)} "
                 f"[{rel.relationship_id}, confidence {rel.confidence:.2f}"
                 f"{', historical' if rel.historical else ''}]"]
        for e in sorted(rel.evidence, key=lambda e: -e.confidence):
            where = f"{e.source_document} ({e.source_file})" if e.source_document else e.source_file
            line = f"  because {where} {e.detail}"
            if e.quote:
                quote = e.quote if len(e.quote) <= 220 else e.quote[:217] + "..."
                line += f' states: "{quote}"'
            if e.chunk_ids:
                line += f"  [chunks: {', '.join(e.chunk_ids[:3])}]"
            lines.append(line)
        return "\n".join(lines)

    def explain_path(self, hops: list[Hop], start: str) -> str:
        lines = [self.label(start)]
        for hop in hops:
            arrow = f"--{hop.relationship.relationship}-->" if hop.direction == OUT else f"<--{hop.relationship.relationship}--"
            lines.append(f"  {arrow} {self.label(hop.node)}  [{hop.relationship.relationship_id}, "
                         f"{hop.relationship.confidence:.2f}]")
        return "\n".join(lines)

    # ---------- summary ----------

    def summary(self) -> dict:
        degree = Counter()
        for rel in self.relationships:
            degree[rel.source_key] += 1
            degree[rel.target_key] += 1
        connected = set(degree)
        isolated = sorted(k for k in self.index.entities if k not in connected)
        return {
            "nodes": len(self.index.entities),
            "edges": len(self.relationships),
            "edges_by_relationship": dict(sorted(Counter(r.relationship for r in self.relationships).items())),
            "historical_edges": sum(1 for r in self.relationships if r.historical),
            "most_connected": [{"key": k, "name": self.label(k), "degree": d} for k, d in degree.most_common(10)],
            "isolated_nodes": isolated,
        }
