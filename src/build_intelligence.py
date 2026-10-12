"""
KORA Phase 3 entry point: turn Phase 2's output into organizational intelligence.

Same shape as Phase 2's src/ingest.py: one function with settings in and a
summary out, so a script, a test, or later the frontend can call it.

    from src.build_intelligence import IntelligenceSettings, run_intelligence
    summary, result = run_intelligence(IntelligenceSettings())

From the terminal (KORA root), after `python -m src.ingest`:
    python -m src.build_intelligence
    python -m src.build_intelligence --no-save

Order matters: each step uses what the previous ones found.
    entities -> rules -> processes -> roles -> topics -> duplicates -> relationships -> graph
    -> conflicts -> outdated -> gaps
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.intelligence.conflicts import detect_conflicts
from src.intelligence.duplicates import find_potential_duplicates
from src.intelligence.entities import EntityIndex, discover_entities
from src.intelligence.gaps import GapReport, detect_gaps
from src.intelligence.graph import KnowledgeGraph
from src.intelligence.loader import Phase2Data, load_phase2
from src.intelligence.outdated import detect_outdated
from src.intelligence.output import write_json
from src.intelligence.processes import ProcessRecord, analyze_processes
from src.intelligence.relationships import KnowledgeRelationship, build_relationships
from src.intelligence.roles import RoleRecord, analyze_roles
from src.intelligence.rules import Rule, extract_rules
from src.intelligence.topics import TopicIntel, discover_topics


@dataclass
class IntelligenceSettings:
    processed_dir: Path = Path("data/processed")     # Phase 2 output (input here)
    chunks_dir: Path = Path("data/chunks")
    raw_dir: Path = Path("data/raw")                 # original text, for structure only
    output_dir: Path = Path("data/intelligence")     # Phase 3 output
    save_output: bool = True


@dataclass
class IntelligenceResult:
    phase2: Phase2Data
    entities: EntityIndex
    rules: list[Rule]
    processes: list[ProcessRecord]
    roles: list[RoleRecord]
    org_chart: dict
    duplicates: dict
    relationships: list[KnowledgeRelationship]
    dangling_relationships: list[dict]
    graph: KnowledgeGraph
    topics: TopicIntel
    conflicts: dict
    outdated: dict
    gaps: GapReport

    def process(self, process_id: str) -> ProcessRecord | None:
        return next((p for p in self.processes if p.process_id == process_id), None)

    def role(self, role_id: str) -> RoleRecord | None:
        return next((r for r in self.roles if r.role_id == role_id), None)

    def rule(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.rule_id == rule_id), None)


@dataclass
class IntelligenceSummary:
    entities_by_type: dict
    rules_by_type: dict
    processes_by_kind: dict
    topics_by_level: dict
    conflicts_by_type: dict
    conflicts_by_status: dict
    corroborations: int
    gaps_by_severity: dict
    archived_documents: int
    relationships: int
    relationships_by_name: dict
    dangling_relationships: int
    potential_duplicates_by_method: dict
    unmatched_names: int
    discovery_notes: int
    collisions: int
    files_written: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_intelligence(settings: IntelligenceSettings | None = None) -> tuple[IntelligenceSummary, IntelligenceResult]:
    settings = settings or IntelligenceSettings()

    phase2 = load_phase2(settings.processed_dir, settings.chunks_dir, settings.raw_dir)
    entities = discover_entities(phase2)
    entities.notes.extend(phase2.load_notes)
    rules = extract_rules(phase2, entities)
    processes = analyze_processes(phase2, entities, rules)
    roles, org_chart = analyze_roles(phase2, entities, processes, rules)
    topics = discover_topics(phase2, entities)                     # adds topic entities
    duplicates = find_potential_duplicates(entities, phase2)       # after roles: org-chart names included
    builder = build_relationships(phase2, entities, processes, roles, rules, topics.topics)
    relationships = builder.results()
    graph = KnowledgeGraph(entities, relationships)
    conflicts = detect_conflicts(phase2, entities, rules, roles, org_chart, duplicates)
    outdated = detect_outdated(phase2, entities, processes, rules, duplicates, conflicts)
    gaps = detect_gaps(phase2, entities, processes, roles, rules, duplicates, org_chart, relationships)

    result = IntelligenceResult(phase2, entities, rules, processes, roles, org_chart, duplicates,
                                relationships, builder.dangling, graph, topics, conflicts, outdated, gaps)

    written = []
    if settings.save_output:
        out = settings.output_dir
        written.append(write_json(entities.to_dict(), out / "entities.json"))
        written.append(write_json(duplicates, out / "potential_duplicates.json"))
        written.append(write_json({
            "count": len(relationships),
            "relationships": [r.to_dict() for r in relationships],
            "dangling": builder.dangling,
        }, out / "relationships.json"))
        written.append(write_json([p.to_dict() for p in processes], out / "processes.json"))
        written.append(write_json({
            "roles": [r.to_dict() for r in roles],
            "approval_rules": [r.to_dict() for r in rules if r.rule_type == "approval"],
            "escalation_rules": [r.to_dict() for r in rules if r.rule_type == "escalation"],
            "org_chart": org_chart,
        }, out / "roles.json"))
        written.append(write_json(graph.summary(), out / "graph_summary.json"))
        written.append(write_json(topics.to_dict(), out / "topics.json"))
        written.append(write_json(conflicts, out / "conflicts.json"))
        written.append(write_json(outdated, out / "outdated_knowledge.json"))
        written.append(write_json(gaps.to_dict(), out / "knowledge_gaps.json"))

    summary = IntelligenceSummary(
        entities_by_type=entities.to_dict()["counts_by_type"],
        rules_by_type=dict(Counter(r.rule_type for r in rules)),
        processes_by_kind=dict(Counter(p.kind for p in processes)),
        topics_by_level=dict(Counter(t.level for t in topics.topics)),
        conflicts_by_type=dict(Counter(c["conflict_type"] for c in conflicts["conflicts"])),
        conflicts_by_status=dict(Counter(c["status"] for c in conflicts["conflicts"])),
        corroborations=len(conflicts["corroborations"]),
        gaps_by_severity=gaps.to_dict()["summary"],
        archived_documents=len(outdated["superseded_documents"]),
        relationships=len(relationships),
        relationships_by_name=dict(sorted(Counter(r.relationship for r in relationships).items())),
        dangling_relationships=len(builder.dangling),
        potential_duplicates_by_method=dict(Counter(d["method"] for d in duplicates["potential_duplicates"])),
        unmatched_names=len(duplicates["unmatched_names"]),
        discovery_notes=len(entities.notes),
        collisions=len(entities.collisions),
        files_written=[str(p) for p in written],
    )
    return summary, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KORA's intelligence layer from Phase 2 output.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/intelligence"))
    parser.add_argument("--no-save", action="store_true", help="Run and report, but don't write files")
    args = parser.parse_args()

    summary, result = run_intelligence(IntelligenceSettings(
        processed_dir=args.processed_dir, chunks_dir=args.chunks_dir, raw_dir=args.raw_dir,
        output_dir=args.output_dir, save_output=not args.no_save,
    ))
    print(json.dumps(summary.to_dict(), indent=2))
    for note in result.entities.notes:
        print(f"[NOTE] {note}")
    for name in result.duplicates["unmatched_names"]:
        print(f"[UNMATCHED] {name['expected_type']} '{name['name']}': {name['reason']}")
    for d in result.dangling_relationships:
        print(f"[DANGLING] {d['source']} --{d['relationship']}--> {d['target']} (missing: {', '.join(d['missing'])})")


if __name__ == "__main__":
    main()
