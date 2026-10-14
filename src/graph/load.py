"""
Load the saved Phase 3 graph (entities.json + relationships.json) back into
Phase 3's own KnowledgeGraph, so every traversal Phase 3 tested (neighbours,
walks, shortest path, explanations) works at runtime without rebuilding.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.intelligence.entities import EntityIndex
from src.intelligence.graph import KnowledgeGraph
from src.intelligence.models import Entity
from src.intelligence.relationships import Evidence, KnowledgeRelationship

ENTITY_FIELDS = {"entity_id", "entity_type", "name", "aliases", "source_documents", "sources",
                 "confidence", "attributes"}
EVIDENCE_FIELDS = {"source_file", "source_document", "detail", "method", "confidence", "chunk_ids",
                   "quote", "historical"}


@dataclass
class LoadedGraph:
    graph: KnowledgeGraph
    dangling: list[dict] = field(default_factory=list)   # links Phase 3 could not attach (reported)


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def relationship_from_dict(r: dict) -> KnowledgeRelationship:
    evidence = [Evidence(**{k: v for k, v in e.items() if k in EVIDENCE_FIELDS}) for e in r["evidence"]]
    rel = KnowledgeRelationship(r["relationship_id"], r["source_id"], r["source_type"], r["relationship"],
                                r["target_id"], r["target_type"], evidence)
    # 'historical' is saved, but endpoint_archived (one of its two causes) is not.
    # It is the only way the saved flag can be true while some evidence is current.
    rel.endpoint_archived = bool(r.get("historical")) and not all(e.historical for e in evidence)
    return rel


def load_graph(intelligence_dir: Path = Path("data/intelligence")) -> LoadedGraph:
    ents = _read(intelligence_dir / "entities.json")
    rels = _read(intelligence_dir / "relationships.json")
    index = EntityIndex()
    for e in ents["entities"] if isinstance(ents, dict) else ents:
        index.add(Entity(**{k: v for k, v in e.items() if k in ENTITY_FIELDS}))
    relationships = [relationship_from_dict(r) for r in rels["relationships"]]
    return LoadedGraph(KnowledgeGraph(index, relationships), list(rels.get("dangling", [])))
