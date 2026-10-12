"""
Topic discovery: what Nexora's knowledge is ABOUT, and which documents discuss what.

Two layers, both explainable:

1. Anchored topics (the main result). Every topic is anchored on something KORA
   already knows, so its name comes from Nexora, not from an algorithm:
      domain    one per department              ('IT / Security', 'Finance')
      product   one per product                 ('NexFlow', category 'Inventory & Operations Software')
      process   one per CURRENT process         ('Customer Onboarding', 'Incident Management')
      program   a document that several others  ('New Joiner Life & Settling In',
                name in their Related header     'Communication Culture & Tone Guidelines')
   A document discusses a topic when:
      it IS the topic's home document                          1.0
      its header or text links to the anchor (Phase 2 / M2)    0.85 / 0.7
      one of its sections is TF-IDF-similar to the home text   0.5, with the score and shared terms
   The similarity route is what finds the FAQ or glossary discussing a topic
   they never name exactly.

2. Emergent themes (exploration). Sections are clustered (agglomerative, cosine)
   with no labels at all. Each cluster is reported with its top terms and the
   anchored topic it overlaps most. A cluster that overlaps nothing is a theme
   no anchor covers, worth a human look.

Why scikit-learn here: TF-IDF turns "which words make this section distinctive"
into numbers, and cosine similarity compares sections without any model or
embedding. Every score can be explained by the shared terms it lists.
"""

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.metadata.relationships import RelationType

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.models import (
    CONF_KEYWORD, CONF_PATTERN, CONF_SIMILARITY, CONF_STRUCTURED,
    DEPARTMENT, DOCUMENT, PROCESS, PRODUCT, TOPIC, Entity,
)
from src.intelligence.normalize import slug
from src.intelligence.text import plain

SIMILARITY_THRESHOLD = 0.30   # section-to-topic cosine; chosen by inspecting real output (see README)
MIN_SECTION_WORDS = 25        # shorter sections are mostly headings or one-liners
CLUSTER_RANGE = range(12, 31, 2)

_VECTORIZER_ARGS = dict(
    stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.3, sublinear_tf=True,
    token_pattern=r"(?u)\b[A-Za-z][A-Za-z-]{2,}\b",     # words only: IDs and numbers carry no topic
)


@dataclass
class Topic:
    topic_id: str
    name: str
    level: str                      # domain | product | process | program
    anchor_key: str                 # the entity this topic is about
    home_documents: list[str]
    keywords: list[str] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)   # {document_id, method, confidence, ...}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Unit:
    document_id: str
    section: str
    chunk_ids: list[str]
    text: str


class _Topics:
    def __init__(self, data: Phase2Data, index: EntityIndex):
        self.data, self.index = data, index
        self.units: list[_Unit] = []
        for doc in sorted(index.of_type(DOCUMENT), key=lambda d: d.entity_id):
            a = doc.attributes
            for s in data.sections(a["header_document_id"], a["version"]):
                text = plain(s.text)
                if len(text.split()) >= MIN_SECTION_WORDS:
                    self.units.append(_Unit(doc.entity_id, s.name, s.chunk_ids, f"{s.name} {text}"))
        self.vectorizer = TfidfVectorizer(**_VECTORIZER_ARGS)
        self.X = self.vectorizer.fit_transform([u.text for u in self.units])
        self.terms = np.array(self.vectorizer.get_feature_names_out())

    # ---------- anchors ----------

    def anchors(self) -> list[Topic]:
        idx = self.index
        topics: list[Topic] = []
        docs = idx.of_type(DOCUMENT)

        for dept in sorted(idx.of_type(DEPARTMENT), key=lambda e: e.entity_id):
            home = sorted(d.entity_id for d in docs if dept.entity_id in d.attributes.get("department_ids", []))
            topics.append(Topic(f"TOPIC-{slug(dept.name)}", dept.name, "domain", dept.key, home))

        for prod in sorted(idx.of_type(PRODUCT), key=lambda e: e.entity_id):
            home = [prod.attributes["document_id"]] if prod.attributes.get("document_id") else []
            topics.append(Topic(f"TOPIC-{slug(prod.name)}", prod.name, "product", prod.key, home))

        for proc in sorted(idx.of_type(PROCESS), key=lambda e: e.entity_id):
            if (proc.attributes.get("status") or "").lower() == "archived":
                continue    # archived versions belong to the current process's topic
            home = [proc.attributes["document_id"]] if proc.attributes.get("document_id") else []
            topics.append(Topic(f"TOPIC-{slug(proc.name)}", proc.name, "process", proc.key, home))

        # Programs: a non-process document named by 2+ other documents' Related headers.
        pointed: dict[str, set[str]] = defaultdict(set)
        for rel in self.data.relationships:
            if rel.relation_type == RelationType.DOCUMENT_RELATED_TO_DOCUMENT:
                src = idx.document_for_relationship(rel)
                if src:
                    pointed[rel.target].add(src)
        roots = {r: m for r, m in pointed.items()
                 if len(m) >= 2 and idx.get(DOCUMENT, r) and not idx.get(PROCESS, r)}
        for root_id, members in sorted(roots.items()):
            root = idx.get(DOCUMENT, root_id)
            # A root that itself points at a bigger root is nested: fold it into the parent.
            # (The full Germany checklist is named by two shorter checklists, but all three
            # belong to the New Joiner program.)
            if any(root_id in other for other_id, other in roots.items() if other_id != root_id):
                continue
            name = re.split(r"\s+[—–-]\s+", root.name)[0].strip()
            topics.append(Topic(f"TOPIC-{slug(name)}", name, "program", root.key, sorted({root_id, *members})))
        return [t for t in topics if t.home_documents]

    # ---------- scoring ----------

    def _profile(self, topic: Topic) -> np.ndarray | None:
        rows = [i for i, u in enumerate(self.units) if u.document_id in topic.home_documents]
        if not rows:
            return None
        centroid = np.asarray(self.X[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm else None

    def _structural_links(self, topic: Topic) -> dict[str, tuple[str, float]]:
        """Documents that already link to the anchor entity (header or text)."""
        anchor_type, anchor_id = topic.anchor_key.split(":", 1)
        links: dict[str, tuple[str, float]] = {}
        wanted = {
            PROCESS: {RelationType.DOCUMENT_RELATED_TO_PROCESS: ("header names the process", CONF_PATTERN),
                      RelationType.DOCUMENT_MENTIONS_PROCESS: ("text mentions the process", CONF_KEYWORD)},
            PRODUCT: {RelationType.DOCUMENT_MENTIONS_PRODUCT: ("text mentions the product", CONF_KEYWORD)},
            DOCUMENT: {RelationType.DOCUMENT_RELATED_TO_DOCUMENT: ("header names the program document", CONF_PATTERN),
                       RelationType.DOCUMENT_MENTIONS_DOCUMENT: ("text mentions the program document", CONF_KEYWORD)},
        }.get(anchor_type, {})
        for rel in self.data.relationships:
            if rel.relation_type in wanted and rel.target == anchor_id:
                doc_id = self.index.document_for_relationship(rel)
                how, conf = wanted[rel.relation_type]
                if doc_id and (doc_id not in links or links[doc_id][1] < conf):
                    links[doc_id] = (how, conf)
        return links

    def score(self, topic: Topic) -> None:
        profile = self._profile(topic)
        if profile is not None:
            topic.keywords = self.terms[profile.argsort()[::-1][:10]].tolist()
        links = self._structural_links(topic)

        best: dict[str, dict] = {}
        if profile is not None:
            sims = self.X @ profile
            for i, u in enumerate(self.units):
                s = float(sims[i])
                if s < SIMILARITY_THRESHOLD or u.document_id in topic.home_documents:
                    continue
                if u.document_id not in best or s > best[u.document_id]["similarity"]:
                    row = self.X[i].toarray().ravel() * profile
                    best[u.document_id] = {
                        "similarity": round(s, 3), "section": u.section, "chunk_ids": u.chunk_ids,
                        "shared_terms": [t for t in self.terms[row.argsort()[::-1][:5]] if row[self.vectorizer.vocabulary_[t]] > 0]}

        for doc_id in topic.home_documents:
            topic.documents.append({"document_id": doc_id, "method": "home_document",
                                    "detail": "the topic's own document", "confidence": CONF_STRUCTURED})
        for doc_id in sorted(set(links) | set(best)):
            if doc_id in topic.home_documents:
                continue
            entry = {"document_id": doc_id}
            if doc_id in links:
                how, conf = links[doc_id]
                entry.update(method="linked", detail=how, confidence=conf)
                if doc_id in best:
                    entry["similarity"] = best[doc_id]
            else:
                entry.update(method="text_similarity", detail=f"section '{best[doc_id]['section']}'",
                             confidence=CONF_SIMILARITY, similarity=best[doc_id])
            topic.documents.append(entry)

    # ---------- emergent themes ----------

    def emergent(self, anchored: list[Topic]) -> dict:
        dense = self.X.toarray()
        best_k, best_score, best_labels = None, -1.0, None
        for k in CLUSTER_RANGE:
            if k >= len(self.units):
                break
            labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(dense).labels_
            score = silhouette_score(self.X, labels, metric="cosine")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        themes = []
        if best_labels is None:
            return {"k": None, "silhouette": None, "themes": themes}
        anchored_docs = {t.topic_id: {d["document_id"] for d in t.documents} for t in anchored}
        for c in sorted(set(best_labels), key=lambda c: -int((best_labels == c).sum())):
            rows = np.where(best_labels == c)[0]
            centroid = np.asarray(self.X[rows].mean(axis=0)).ravel()
            docs = sorted({self.units[i].document_id for i in rows})
            overlap = max(((len(set(docs) & d) / len(set(docs) | d), tid) for tid, d in anchored_docs.items()),
                          default=(0.0, None))
            themes.append({
                "theme_id": f"THEME-{len(themes) + 1:02d}",
                "top_terms": self.terms[centroid.argsort()[::-1][:8]].tolist(),
                "sections": [{"document_id": self.units[i].document_id, "section": self.units[i].section} for i in rows],
                "documents": docs,
                "closest_anchored_topic": overlap[1],
                "overlap": round(overlap[0], 2),
                "unanchored": overlap[0] < 0.3,
            })
        return {"k": best_k, "silhouette": round(float(best_score), 3),
                "note": "Unsupervised clusters of sections; labels are top terms, not names. "
                        "Low silhouette is expected for ~50 short documents: use for exploration only.",
                "themes": themes}

    def document_keywords(self) -> dict[str, list[str]]:
        by_doc: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(self.units):
            by_doc[u.document_id].append(i)
        out = {}
        for doc_id, rows in sorted(by_doc.items()):
            v = np.asarray(self.X[rows].mean(axis=0)).ravel()
            out[doc_id] = self.terms[v.argsort()[::-1][:8]].tolist()
        return out


@dataclass
class TopicIntel:
    topics: list[Topic]
    emergent: dict
    document_keywords: dict
    documents_without_topic: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        per_doc = Counter(d["document_id"] for t in self.topics for d in t.documents)
        return {
            "method": {"similarity_threshold": SIMILARITY_THRESHOLD, "min_section_words": MIN_SECTION_WORDS,
                       "vectorizer": {k: v for k, v in _VECTORIZER_ARGS.items() if k != "token_pattern"}},
            "topics": [t.to_dict() for t in self.topics],
            "documents_without_topic": self.documents_without_topic,
            "topics_per_document": dict(sorted(per_doc.items())),
            "document_keywords": self.document_keywords,
            "emergent_themes": self.emergent,
        }


def discover_topics(data: Phase2Data, index: EntityIndex) -> TopicIntel:
    t = _Topics(data, index)
    topics = t.anchors()
    for topic in topics:
        t.score(topic)
        index.add(Entity(
            entity_id=topic.topic_id, entity_type=TOPIC, name=topic.name,
            source_documents=topic.home_documents,
            sources=[f"anchored on {topic.anchor_key} ({topic.level})"],
            confidence=CONF_STRUCTURED,
            attributes={"level": topic.level, "anchor": topic.anchor_key, "keywords": topic.keywords},
        ))
    covered = {d["document_id"] for topic in topics for d in topic.documents}
    without = sorted(d.entity_id for d in index.of_type(DOCUMENT) if d.entity_id not in covered)
    return TopicIntel(topics, t.emergent(topics), t.document_keywords(), without)
