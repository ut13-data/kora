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
    entities -> rules -> processes -> roles -> topics -> germany -> communication -> duplicates
    -> relationships -> graph -> conflicts -> outdated -> gaps -> employee context

Two INPUT files are read (never written) from data/context/:
    checklist_progress.json        what employees reported as done (starts empty)
    communication_preferences.json individual communication preferences (starts empty)
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from src.intelligence.communication import CommunicationIntel, build_communication
from src.intelligence.conflicts import detect_conflicts
from src.intelligence.duplicates import find_potential_duplicates
from src.intelligence.entities import EntityIndex, discover_entities
from src.intelligence.employee_context import build_employee_context
from src.intelligence.gaps import GapReport, add_gaps, detect_gaps
from src.intelligence.germany import GermanyCare, build_germany_care
from src.intelligence.graph import KnowledgeGraph
from src.intelligence.loader import Phase2Data, load_phase2
from src.intelligence.outdated import detect_outdated
from src.intelligence.output import _to_jsonable, write_json
from src.intelligence.questions import answer_all
from src.intelligence.report import build_report
from src.intelligence.validation import validate_outputs
from src.intelligence.processes import ProcessRecord, analyze_processes
from src.intelligence.models import CHECKLIST, CONF_STRUCTURED, EMPLOYEE
from src.intelligence.relationships import Evidence, KnowledgeRelationship, build_relationships
from src.intelligence.roles import RoleRecord, analyze_roles
from src.intelligence.rules import Rule, extract_rules
from src.intelligence.topics import TopicIntel, discover_topics


# Joining stages are computed relative to this date. Nobody in employees.csv joined
# within 90 days of 2026, so the demo date puts real relocating employees mid-program:
# EMP-0085 (Munich) is in week 2 and EMP-0023 (Munich) in month 2. Override with --as-of.
DEMO_AS_OF = date(2025, 4, 22)


@dataclass
class IntelligenceSettings:
    processed_dir: Path = Path("data/processed")     # Phase 2 output (input here)
    chunks_dir: Path = Path("data/chunks")
    raw_dir: Path = Path("data/raw")                 # original text, for structure only
    context_dir: Path = Path("data/context")         # inputs: checklist progress, communication preferences
    output_dir: Path = Path("data/intelligence")     # Phase 3 output
    as_of: date = DEMO_AS_OF
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
    germany: GermanyCare
    communication: CommunicationIntel
    employee_context: dict
    validation: dict = field(default_factory=dict)
    answers: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)      # every output exactly as written (JSON shape)

    def employee(self, employee_id: str) -> dict | None:
        return next((e for e in self.employee_context["employees"] if e["employee_id"] == employee_id), None)

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
    as_of: str
    germany_employees_by_status: dict
    communication_profiles: int
    new_joiners: int
    validation: dict
    questions_answered: str
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
    germany = build_germany_care(phase2, entities, settings.as_of, settings.context_dir / "checklist_progress.json")
    communication = build_communication(phase2, entities, settings.context_dir / "communication_preferences.json")
    duplicates = find_potential_duplicates(entities, phase2)       # after roles: org-chart names included
    builder = build_relationships(phase2, entities, processes, roles, rules, topics.topics,
                                  _germany_links(entities, germany))
    relationships = builder.results()
    graph = KnowledgeGraph(entities, relationships)
    conflicts = detect_conflicts(phase2, entities, rules, roles, org_chart, duplicates)
    outdated = detect_outdated(phase2, entities, processes, rules, duplicates, conflicts)
    gaps = detect_gaps(phase2, entities, processes, roles, rules, duplicates, org_chart, relationships)
    add_gaps(gaps, germany.gaps, "germany_checklist",
             "Germany checklist items have a Nexora contact (locally), a Nexora document or an official source")
    employee_context = build_employee_context(entities, processes, roles, rules, relationships, germany,
                                              communication, conflicts, settings.as_of)

    result = IntelligenceResult(phase2, entities, rules, processes, roles, org_chart, duplicates,
                                relationships, builder.dangling, graph, topics, conflicts, outdated, gaps,
                                germany, communication, employee_context)

    # Every output, exactly as it will be written. Validation reads these JSON-shaped
    # copies (not the Python objects), so a serialisation bug is caught too.
    outputs = {
        "entities": entities.to_dict(),
        "potential_duplicates": duplicates,
        "relationships": {"count": len(relationships), "relationships": [r.to_dict() for r in relationships],
                          "dangling": builder.dangling},
        "processes": [p.to_dict() for p in processes],
        "roles": {"roles": [r.to_dict() for r in roles],
                  "approval_rules": [r.to_dict() for r in rules if r.rule_type == "approval"],
                  "escalation_rules": [r.to_dict() for r in rules if r.rule_type == "escalation"],
                  "org_chart": org_chart},
        "graph_summary": graph.summary(),
        "topics": topics.to_dict(),
        "conflicts": conflicts,
        "outdated_knowledge": outdated,
        "knowledge_gaps": gaps.to_dict(),
        "germany_checklists": germany.to_dict(entities),
        "communication_guidance": communication.to_dict(entities),
        "employee_context": employee_context,
    }
    outputs = json.loads(json.dumps(outputs, default=_to_jsonable))
    validation = validate_outputs(outputs, phase2.chunks)
    result.outputs = outputs
    result.validation = validation
    result.answers = answer_all(result)

    written = []
    if settings.save_output:
        out = settings.output_dir
        for stem, data in outputs.items():
            written.append(write_json(data, out / f"{stem}.json"))
        written.append(write_json(validation, out / "intelligence_validation.json"))
        written.append(write_json(result.answers, out / "test_questions.json"))

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
        as_of=settings.as_of.isoformat(),
        germany_employees_by_status=dict(Counter(germany.eligibility(e, entities)["status"]
                                                 for e in entities.of_type(EMPLOYEE))),
        communication_profiles=len(communication.profiles),
        new_joiners=len(employee_context["new_joiners"]),
        validation=validation["summary"],
        questions_answered=f"{sum(a['status'] == 'answered' for a in result.answers)}/{len(result.answers)}",
        relationships=len(relationships),
        relationships_by_name=dict(sorted(Counter(r.relationship for r in relationships).items())),
        dangling_relationships=len(builder.dangling),
        potential_duplicates_by_method=dict(Counter(d["method"] for d in duplicates["potential_duplicates"])),
        unmatched_names=len(duplicates["unmatched_names"]),
        discovery_notes=len(entities.notes),
        collisions=len(entities.collisions),
        files_written=[str(p) for p in written],
    )
    if settings.save_output:
        report_path = settings.output_dir / "knowledge_report.md"
        report_path.write_text(build_report(result, summary, validation, result.answers), encoding="utf-8")
        summary.files_written.append(str(report_path))
    return summary, result


def _germany_links(index, germany: GermanyCare) -> list[tuple]:
    """Employee -> eligible_for -> Checklist, for everyone relocating to a German office."""
    links = []
    for emp in index.of_type(EMPLOYEE):
        elig = germany.eligibility(emp, index)
        if elig["status"] in ("active", "pre_start", "past_program_window"):
            for checklist in index.of_type(CHECKLIST):
                links.append((EMPLOYEE, emp.entity_id, "eligible_for", CHECKLIST, checklist.entity_id,
                              Evidence("employees.csv", None, "columns location_id + international_relocation_germany",
                                       "table", CONF_STRUCTURED)))
    return links


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KORA's intelligence layer from Phase 2 output.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/intelligence"))
    parser.add_argument("--context-dir", type=Path, default=Path("data/context"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEMO_AS_OF,
                        help=f"Date joining stages are computed from (default {DEMO_AS_OF.isoformat()})")
    parser.add_argument("--no-save", action="store_true", help="Run and report, but don't write files")
    args = parser.parse_args()

    summary, result = run_intelligence(IntelligenceSettings(
        processed_dir=args.processed_dir, chunks_dir=args.chunks_dir, raw_dir=args.raw_dir,
        context_dir=args.context_dir, output_dir=args.output_dir, as_of=args.as_of, save_output=not args.no_save,
    ))
    print(json.dumps(summary.to_dict(), indent=2))
    for note in result.entities.notes:
        print(f"[NOTE] {note}")
    for name in result.duplicates["unmatched_names"]:
        print(f"[UNMATCHED] {name['expected_type']} '{name['name']}': {name['reason']}")
    for d in result.dangling_relationships:
        print(f"[DANGLING] {d['source']} --{d['relationship']}--> {d['target']} (missing: {', '.join(d['missing'])})")
    for check in result.validation["checks"]:
        if check["status"] != "pass":
            print(f"[{check['status'].upper()}] {check['check']}: {check['count']} -- {check['description']}")


if __name__ == "__main__":
    main()
