# KORA: Knowledge Object Retrieval Agent

KORA turns a company's internal knowledge (policies, processes, product docs, org data) into a clean, structured dataset, and then into an **understanding of the organization** behind it: who does what, who approves what, what conflicts, what is missing, and what matters to a specific employee. All of it is built so that later retrieval and an AI assistant can answer questions with sources.

The source data is **Nexora Systems**, a synthetic mid-sized B2B software company (53 Markdown documents, 7 CSV tables).

## Status

| Phase | What | Status |
|---|---|---|
| 2 | Ingestion: extract, clean, chunk, metadata, validate | complete |
| 3 | Knowledge intelligence: entities, relationships, processes, topics, quality, people | complete |
| next | Retrieval and reasoning on top of the intelligence layer | planned |

Phase 3 deliberately uses **no RAG, embeddings, vector database, LLM or external AI service**: only Python, regular expressions, and scikit-learn (TF-IDF) where it genuinely helps.

## Quick start

Requires Python 3.10+ (developed on 3.14).

```
python -m venv venvkora
venvkora\Scripts\activate
pip install -r requirements.txt
```

Knowledge files live in `data/raw/`. Then:

```
python -m src.ingest                      # Phase 2: documents -> chunks + metadata
python -m src.build_intelligence          # Phase 3: -> data/intelligence/*.json + knowledge_report.md
python -m pytest test/ -v                 # 136 tests

python -m src.explain EMP-0077 PROD-NEXFLOW          # how two things connect, with evidence
python -m src.explain process:PROC-IMPL-001 --rel approves
python -m src.employee EMP-0085                      # what is relevant to one employee
python -m src.employee --hypothetical                # "a new Data Analyst in Munich, week 2"
```

From Python:

```python
from src.build_intelligence import IntelligenceSettings, run_intelligence

summary, result = run_intelligence(IntelligenceSettings())
result.graph.explain(result.relationships[0])
result.communication.present("Please complete the security training by Friday.", "COMM-PROFILE-GERMANY", "direct")
result.employee("EMP-0085")
```

---

## Phase 2: Ingestion

```
Upload -> Extract -> Clean -> Identify metadata -> Chunk -> Relationships -> Validate -> Store
```

| Stage | What it does |
|---|---|
| Upload / Input | Identifies file name, type, size, source path, ingestion timestamp. Rejects unsupported types. |
| Extraction | Markdown, TXT, PDF, DOCX to plain text. CSV stays tabular (one dict per row). |
| Cleaning | Removes whitespace noise and encoding artifacts. Never changes words or numbers. |
| Document identification | Reads the `**Label:** value` header. Missing fields are `unknown`, never invented. |
| Chunking | Structure-aware: Markdown headings, then size (about 1,200 characters). Each chunk keeps `section` and `parent_id`. Stable IDs such as `PROC-HR-001_v3.0_002`. |
| Entity registry | One lookup of every department, role, process, product, location, employee and document from the CSVs. |
| Relationships | From headers, CSV tables, text discovery and location links. Every relationship records its evidence. |
| Validation | Reports problems, never silently fixes them. |

Phase 2 output: `data/processed/` (documents, relationships, tables, validation report) and `data/chunks/chunks.jsonl`.

Settings: `include_archived` (default `True`: employees can see what a rule was before and what it is now), `chunk_size` (1200), `chunk_overlap` (100).

---

## Phase 3: Knowledge intelligence

Phase 3 reads Phase 2's **saved output files**, not its code paths, so the phases stay independent. Structure (steps, role paragraphs, the org chart) is parsed from the original text through Phase 2's own extractor, because chunks strip the indentation that marks structure; every finding still cites chunk IDs.

```
entities -> rules -> processes -> roles -> topics -> germany -> communication -> duplicates
   -> relationships -> graph -> conflicts -> outdated -> gaps -> employee context
   -> validation -> test questions -> knowledge_report.md
```

### Modules (`src/intelligence/`)

| Module | What it discovers |
|---|---|
| `entities.py` | 13 entity types (employee, department, role, process, product, customer, policy, document, location, topic, checklist, terminology, audience), each with a stable ID, aliases, sources and confidence |
| `duplicates.py` | Possible duplicates and aliases, each with method, reason and confidence. **Never merges.** |
| `rules.py` | Approval and escalation rules: who approves what, at which threshold, who receives escalations |
| `processes.py` | Per process: kind (procedure/policy/guideline), owner, department, participants, steps, inputs/outputs (only when stated), handoffs, policies, products, known gaps |
| `roles.py` | Per role: responsibilities, reporting lines (org chart), processes, approvals, escalations, documents |
| `relationships.py` | One relationship layer (`Entity -> relationship -> Entity`), every link with evidence |
| `graph.py` | Lightweight graph: neighbours, fixed-path walks, shortest path, plain-English explanations |
| `topics.py` | 37 anchored topics (TF-IDF scored) plus emergent themes from clustering |
| `conflicts.py` | Disagreements between sources, with metadata signals. **Never picks a winner.** |
| `outdated.py` | Archived documents, what replaced them, and what changed step by step |
| `gaps.py` | Missing or incomplete knowledge, with severity; clean checks are reported too |
| `germany.py` | Germany settling-in checklist: items, pace, dependencies, official-guidance caveats, per-employee plan |
| `communication.py` | Four-layer framework, tone profiles, hedged-language check, presentation contract for a future UI |
| `employee_context.py` | Everything relevant to one employee, each item with the reason it is relevant |
| `validation.py` | Checks every output; FAIL = KORA bug, WARN = source-data issue |
| `questions.py` | 26 test questions answered from the intelligence layer, with sources |
| `report.py` | `knowledge_report.md` |

### Inputs (`data/context/`, read but never written)

| File | Purpose |
|---|---|
| `checklist_progress.json` | What employees **reported** as done on the Germany checklist. Starts empty: nothing is ever marked complete by KORA. |
| `communication_preferences.json` | Individual communication preferences, set by the person. Always override the location profile. Starts empty. |

Joining stages are computed relative to `as_of` (default **2025-04-22**, so that real relocating employees are mid-program: EMP-0085 in week 2, EMP-0023 in month 2). Override with `--as-of YYYY-MM-DD`.

### Outputs (`data/intelligence/`)

`entities.json`, `potential_duplicates.json`, `relationships.json`, `processes.json`, `roles.json`, `graph_summary.json`, `topics.json`, `conflicts.json`, `outdated_knowledge.json`, `knowledge_gaps.json`, `germany_checklists.json`, `communication_guidance.json`, `employee_context.json`, `intelligence_validation.json`, `test_questions.json`, `knowledge_report.md`.

### Confidence scale

| Value | Means |
|---|---|
| 1.0 | Read from a CSV column or document header field |
| 0.95 | A Nexora document explicitly states it (e.g. glossary "Same role.") |
| 0.85 | A fixed text pattern in a known section (a step, a role paragraph, the org chart) |
| 0.7 | Name or keyword match in free text |
| 0.5 | Similarity only: a suggestion for a human to review |

### Design decisions

- **Deterministic first.** Every inference has evidence, a source and a confidence; machine learning is used only where it adds something (TF-IDF for topics and for finding similar rule statements).
- **Report, never decide.** Conflicts show both sides and the signals (active vs archived, effective date, version lineage, declared supersession); the status summarises signals and is never a verdict.
- **Never invent.** Unknown names are reported, not mapped to the "closest" entity; inputs/outputs are only recorded when stated; no deadline, date or legal rule is generated.
- **Thresholds tuned on real output** (`conflicts.py`): numeric facts need TF-IDF similarity ≥ 0.6 (the real conflict scored 0.88; different tiers of one rule scored ~0.5). Approval rules are compared on their *subject* (role names and approval words removed, ≥ 0.4) and tiers with non-overlapping ranges are recognised as tiers.
- **Culture is a hypothesis, never a fact about a person.** Profiles must use hedged language (checked); individual preference overrides; nothing is inferred from name or nationality.
- **History is kept.** Archived documents stay in the graph, flagged `historical`; `current_only=True` hides them when needed.

### What Phase 3 found in Nexora

- **Conflicts:** security training 10 working days (current) vs 30 days (archived), declared by both documents; "four office locations" in three documents vs five in `locations.csv`; the org chart places a department head (Implementation Manager) under another head, contradicting its own rule; one document ID used by two files.
- **Gaps (high):** the three declared Known Gaps (KG-01, KG-02, and the proposal-change gap). **Medium:** NexCRM and NexAuto lack implementation guidance; the CEO and "VP People & Operations" own documents but are not roles; Level 3 customer escalation targets an undefined executive.
- **Corroborations:** FAQ answers match the processes; stated headcounts match `employees.csv`; "Project Lead" (v1.0) and "Implementation Manager" (v2.0) resolve to the same approver.
- **Germany care:** 15 items; address registration, health insurance and tax keep the "check the current official guidance" caveat; Berlin has no local contact for housing and emergency information.

## Project structure

```
KORA/
├── data/
│   ├── raw/              # source files (never modified)
│   ├── processed/        # Phase 2 JSON output
│   ├── chunks/           # chunks.jsonl
│   ├── context/          # Phase 3 INPUTS: checklist progress, communication preferences
│   └── intelligence/     # Phase 3 output + knowledge_report.md
├── src/
│   ├── ingest.py               # Phase 2 entry point
│   ├── build_intelligence.py   # Phase 3 entry point
│   ├── explain.py              # explore the graph from the terminal
│   ├── employee.py             # one employee's context from the terminal
│   ├── pipeline.py, ingestion/, extraction/, cleaning/, chunking/, metadata/, validation/, storage/
│   └── intelligence/           # Phase 3 modules (one file per concern)
├── test/                       # pytest: Phase 2 + Phase 3 (136 tests)
├── requirements.txt
└── README.md
```

## Known data findings (reported, not fixed)

These are issues in the Nexora source data, not in KORA:

1. `PROC-IMPL-001` exists as v2.0 (Active) and v1.0 (Archived), kept on purpose as version history.
2. The archived onboarding file's header says `PROC-IMPL-001`, but `documents.csv` lists it as `PROC-IMPL-001-V1`.
3. 14 employees have no `manager_id`; 12 head a department, 2 are Office Managers.
4. Document owners "Chief Executive Officer" and "VP People & Operations" are not roles in `roles.csv`.

## Next phase

Retrieval and reasoning on top of this layer: retrieve chunks, then use entities, relationships, rules, conflicts and employee context to answer with sources, the right version, and the right tone.
