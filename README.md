# KORA: Knowledge Object Retrieval Agent

KORA turns a company's internal knowledge (policies, processes, product docs, org data) into a clean, structured dataset, then into an **understanding of the organization** behind it (who does what, who approves what, what conflicts, what is missing), and then **answers questions from it with sources**: every answer line is quoted word-for-word from a cited document, and when the documents don't say something, KORA says so instead of guessing.

The source data is **Nexora Systems**, a synthetic mid-sized B2B software company (53 Markdown documents, 7 CSV tables).

## Status

| Phase | What | Status |
|---|---|---|
| 2 | Ingestion: extract, clean, chunk, metadata, validate | complete |
| 3 | Knowledge intelligence: entities, relationships, processes, topics, quality, people | complete |
| 4 | Retrieval, reasoning, cited answers, validation, access control, evaluation | complete |
| 5 | Knowledge graph reasoning, UI and dashboards | in progress (M0-M3 done) |

KORA uses **no LLM, vector database or external AI service**: only Python, regular expressions and scikit-learn (TF-IDF). Meaning-based search with a small local embedding model exists as an option and is **off** by default (see Phase 4).

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
python -m pytest test/ -q                 # 420 tests

python -m src.ask "Who approves purchases above 10000 euros?"     # Phase 4: ask, with sources
python -m src.ask "How soon is a business trip paid back?" --why  # ... and why KORA chose its status
python -m src.ask "What are the payment terms for vendor invoices?" --as EMP-0113   # as an employee
python -m src.ask                                                 # interactive (loads once)
python -m src.index --status                                      # is the index current?
python -m src.evaluation.run_eval                                 # evaluation vs a plain TF-IDF baseline
python -m src.evaluation.baseline                                 # every question set vs the frozen Phase 4 results

python -m src.explain EMP-0077 PROD-NEXFLOW          # how two things connect, with evidence
python -m src.explain process:PROC-IMPL-001 --rel approves
python -m src.explain process:PROC-FIN-002 --as EMP-0040   # as one employee may see it
python -m src.graph.quality                          # graph health -> reports/graph_quality.md
python -m src.graph.ask "Which teams are involved in customer onboarding?"   # graph-only answer (M2)
python -m src.graph.ask --dev                        # the graph on its development questions
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

---

## Phase 4: Retrieval, reasoning and cited answers

```
Question -> Route (catalogue) -> Retrieve -> Reason -> Answer -> Cite -> Validate
```

| Step | What it does | Code |
|---|---|---|
| Index check | Before every question: has `data/raw` changed (size/mtime, then SHA-256)? If so, Phase 2 + 3 are rebuilt into a temporary folder and swapped in only on success, so a new file is answerable on the next question and a failed rebuild keeps the old index. | `src/index.py` |
| Route | ASRS-style navigation over Phase 3's catalogue: topics, roles, locations, documents named by ID, intents (approval, escalation, conflict, gap, version, document listing), Germany relocation and communication programs, archived handling. Gives candidate documents, conflicts to show, and a reason for every choice. | `src/retrieval/router.py` |
| Retrieve | Tiered search: conflict sides, named documents, routed candidates (capped at 3 of 5 slots), then 2 slots always reserved for a whole-corpus TF-IDF search so a routing mistake can't hide the answer, then documents that only mention a named ID. | `src/retrieval/search.py`, `retrieve.py` |
| Reason | Phase 3 rules (approval / escalation), conflict sides (never a winner), version changes, Germany checklist items for the asked time window, the right country tone profile, declared gaps. | `src/retrieval/answer.py` |
| Answer | Short: 2-3 quoted lines. Status: **answered**, **partial** (with the uncovered words named), **conflict**, or **not_found**. | `answer.py` |
| Cite | Every line carries a numbered source: document ID, version, status, effective date, section, chunk, file. | `answer.py` |
| Validate | At runtime every line is checked word-for-word against its cited chunk; a line that fails is removed and recorded. Notes flag archived sources, unknown metadata, known gaps, older versions and relocation disclaimers. | `answer.py` |

### Precision first: when KORA refuses

KORA prefers "not found" to a wrong answer. It refuses when:
- the question uses a word **no document ever uses** ("pets", "Madrid", "overtime"); the "why" view names the word;
- the evidence is weak **and** the question's main words are not together in one quoted line (stops answers stitched from unrelated sentences, e.g. "salary structure");
- none of the main words appear in the closest sources.

A word the documents use elsewhere but not in the quoted source ("contractors") makes the answer **partial**, with the word named.

### Configuration (`config/kora_config.json`)

Company-specific behaviour lives here, validated on load (every problem listed at once); Phase 5's UI will edit it.

| Section | Purpose |
|---|---|
| `thresholds` | Every tunable number, each with an allowed range |
| `synonyms` | **Company word list**: everyday phrase -> the words the documents use (`"trip": ["travel"]`). Used for search and coverage, shown in the answer when used. This is how everyday wording is taught to KORA. |
| `access` | **Access control** (M7): document ID pattern -> groups allowed to read it (`all`, `manager`, `dept:<id>`, `loc:<id>`, `role:<id>`, `emp:<id>`, from the employees table). Documents without a rule are open. A hidden document is never searched, quoted, cited or named; the user gets the same "not found" as for undocumented things. The two Nexora rules are an example. |
| `relocation`, `communication` | Germany new-joiner program and country tone profiles |
| `embeddings` | Optional meaning-based search (`sentence-transformers`, all-MiniLM-L6-v2). **Off**: it improved ranking but not answers on unseen questions, and added PyTorch and a ~8 s start-up. |
| `gap_severities`, `intent_patterns` | Which known gaps to show; extra intent patterns |

### Every answer also carries

- `decision`: the "why" (which rule decided, scores and thresholds, unknown words, word-list matches) -- `--why` in the CLI;
- `timings`: per step for this answer (index check, route, search, compose) and KORA's start-up per step;
- `user`: who asked. Every answer is appended to `data/logs/answers.jsonl` (for Phase 5 dashboards).

### Evaluation (`src/evaluation/`, `reports/`)

Questions with document-level gold labels; KORA vs a plain TF-IDF baseline (no router, no answer layer). The development set was used while building; held-out sets were written before the code they test and run once.

| Set | n | Baseline: right doc in top 5 | KORA: right doc in top 5 | KORA correct status | Faithful |
|---|---|---|---|---|---|
| Development | 41 | 0.90 | 1.00 | 0.98 | 1.00 |
| Held-out v1 / v2 / v3 (builder-written) | 30 | 0.96 avg | 0.96 avg | 0.70 / 0.80 / 0.70 | 1.00 |
| Held-out v4 (everyday wording, builder) | 10 | 0.57 | 0.57 | 0.30 | 1.00 |
| Held-out v5 (written by UT, independent) | 10 | 0.50 | 0.50 | 0.80 | 1.00 |

Faithfulness is 1.00 everywhere (enforced at runtime). Answers take ~7 ms (p95 ~10 ms); start-up ~0.4 s. The numbers above are after all fixes; the first, untouched run of each held-out set and every fix made after it are recorded honestly in `reports/phase4_m5_findings.md`.

**Known limits.** KORA understands the documents' own words plus the company word list: everyday wording it hasn't been taught is refused (safe, but not helpful; v4). Homonyms can slip through as *partial*: "notice period" (HR) matched an engineering "notice period"; "stock" (shares vs inventory). Thresholds were tuned on one synthetic company. Answers are quotes, not prose.

## Phase 5: Knowledge graph reasoning, UI and dashboards (in progress)

Plan and decisions: `claude/phase5_plan.md` in the KORA project. Stack: FastAPI + plain HTML (no Node). Relationships that come from a CSV row are cited as **Record** citations. Admins are an explicit list in the config.

### M0: honest before/after numbers (done)

| What | Why |
|---|---|
| **Frozen baseline** `data/evaluation/baseline_phase4.json`; `python -m src.evaluation.baseline` compares today's KORA with it on every question set and lists FIXED / BROKEN / CHANGED questions (exit code 1 if anything broke) | Every Phase 5 change is measured against the real Phase 4 results, not memory |
| **Relationship question sets** `graph_dev_questions.json` (22, used while building) and `graph_heldout_questions.json` (12, run once after M3), written before any graph-answer code | Proves whether the graph helps, without tuning to the test |
| **New scores**: `names_found` (does the answer name the expected roles / processes?), `no_leak` (forbidden names never visible), and `as_user` questions | "Answered" alone overstates: Phase 4 answers 6 graph questions with the right status but names none of the expected owners or teams |
| **Answer log schema 2**: every record has `schema`, `kind` (`answer` / `error`) and `channel` (`cli` / `ui` / `api` / `python`); crashes are logged as `error` and still raised. Old records read as kind `answer`, channel `unknown` | Dashboards can separate a crash from an honest "not found", and real use from testing |
| **Stable rebuilds**: equal TF-IDF scores are ranked A-Z; context paths use `/` | The same files gave slightly different `data/intelligence` JSON on Windows and Linux |

Phase 4 behaviour is unchanged: every question in every set gives the same status and citations as before M0.

### M1: a graph KORA can trust, per user (done)

New package `src/graph/` reads the saved Phase 3 graph (no rebuild) and adds:

| Module | What it does |
|---|---|
| `load.py` | `entities.json` + `relationships.json` -> Phase 3's `KnowledgeGraph` (all traversals work as before) |
| `schema.py` | The 49 valid link shapes (`role -approves-> process`, ...). A new or misspelled shape is reported, not used |
| `trace.py` | Re-checks every piece of evidence against its source and gives each link a **provenance tier**: **quote** (sentence found in its chunk; a table rule is cited as the verbatim row), **header** (header line in the file names it), **record** (CSV row + column holds it), **mention** (named in the same text), **derived** (KORA's own analysis). Only quote / header / record are *documented*: the only links an answer may state as fact |
| `view.py` | `GraphView.for_user(graph, user, config)`: the graph as one employee may see it. Restricted documents disappear together with the processes and topics they describe, every link to them, the evidence they provide and the counts. Colleagues see only directory fields; the person and admins see the rest |
| `quality.py` | `python -m src.graph.quality [--as EMP-x]`: tiers per relationship, unverified evidence, unknown shapes, dangling links, processes without owner or participants, open conflicts, superseded documents, gaps. Every number has a written definition (for the dashboards) |

New config sections: `admins.employees` (dashboards, settings, colleagues' private details; **no** extra document access) and `person_visibility.colleague_fields` (**provisional**: to be decided after testing employees of every level in the UI).

Nexora results: all 1,974 evidence items verify (tests prove the checks fail when a quote, header line or table value changes). 63% of links are documented: quote 114, header 428, record 688; mention 547, derived 171. An Engineering employee sees 350 of 354 entities and 1,852 of 1,948 links.

Findings for M2:
- A relationship's *name* can say more than its evidence. 10 of 11 `governed_by` links come from a "Related Processes" header line: the document lists the policy as related, it doesn't say "governs". All 27 `eligible_for` links are computed (location + relocation flag). Answers must be phrased from the evidence, not from the link name.
- `PROC-HR-003` (Internal Communication Guidelines) has no participating roles: it is addressed to all employees.

### M2: relationship questions answered from the graph (done)

`src/graph/resolve.py` finds the entities a question names (visible ones only; stems, aliases, longest name first; ambiguous words like "onboarding" alone match nothing). `src/graph/query.py` recognises the intent and answers:

| Kind | Example | Built from |
|---|---|---|
| involved | Which teams are involved in customer onboarding? | roles with a step (quote), owners, approvers, header roles; grouped by the role's department |
| owner | Who owns the procurement process? | `owns` (company record / header), owning department |
| policies | Which policies govern incident management? | header cross-references, phrased "X lists Y as related" (never "governs") |
| impact | What could be affected if NexConnect changes? | `uses` / `triggers` (quotes) as dependencies; header cross-references only as "possibly related"; always notes that no document describes the effect of a change |
| delay | Why is customer onboarding delayed? | owner, what feeds it, declared known gaps, open conflicts. **Always partial**: KORA has no live status, so it never names the cause of one case |
| next | Which process starts when a deal is closed? | `triggers` links, matched on their quoted sentence when no process is named |
| department / role / person / uses / which department / head of the owning department | | records and headers |
| approver (old versions only) | Who gave final approval before 2025? | the archived version's quote, flagged as history, with the old role name |

Every statement comes from the link's verified evidence, its verb matches that evidence, and only documented links (quote / header / record) are facts. The graph returns **nothing** (Phase 4 answers as before) when it doesn't recognise the intent, can't find a visible entity, or has no documented fact. It deliberately leaves to Phase 4: approval questions (its rule engine knows amounts and tiers), "what does a <role> do" (Phase 4 quotes the role description), and anything with "after" that isn't about a next process.

Development results (graph alone, 22 development questions): right status 17/22 (the other 5 are by design Phase 4 questions: access refusal, privacy, controls, a plain approval) and 29/30 expected names stated, against 12% names for Phase 4 alone. Of the 81 Phase 4 questions the graph takes exactly one (Q22, an old-version approval, answered correctly). About 0.6 ms per graph answer.

### M3: graph answers in `Kora.ask` (done)

Relationship questions now get graph answers through the normal flow (CLI `python -m src.ask`, and the UI later), with the same citations, runtime verification, access control, `--why`, timings and log. Phase 4 still runs for every question. Its answer is kept when the graph has nothing documented, and its conflict sides are kept first if it found a conflict. New citation kinds: **document header** and **company record**, both re-checked at answer time. Switch: `"graph": {"enabled": true}` in the config.

| Set | Status before -> after | Expected names before -> after |
|---|---|---|
| Phase 4 development + held-out v1-v5 (91) | unchanged, every question | n/a |
| Graph development (22) | 0.59 -> 1.00 | 0.12 -> 1.00 |
| Graph held-out (12; 9 unseen: 9/9) | 0.75 -> 1.00 | 0.19 -> 1.00 |

No leaks, faithfulness 1.00, latency unchanged (~6 ms p50). Full account, including what changed after the held-out run and the limits: `reports/phase5_m3_findings.md`.

**Honesty note:** while building M2 I (Claude) looked at three held-out graph questions by mistake: H02 (ran it), H04 and H09 (copied into resolver tests). They are marked `seen_in_development` with a note and will be reported separately; 9 held-out questions remain unseen. Both graph sets were written by the builder, so they are not independent of the design either.

Phase 4 baseline on the relationship sets (development / held-out): correct status 0.59 / 0.75, expected names stated 0.12 / 0.19, no leaks, faithfulness 1.00.

## Project structure

```
KORA/
├── data/
│   ├── raw/              # source files (never modified)
│   ├── processed/        # Phase 2 JSON output
│   ├── chunks/           # chunks.jsonl
│   ├── context/          # Phase 3 INPUTS: checklist progress, communication preferences
│   ├── intelligence/     # Phase 3 output + knowledge_report.md
│   ├── evaluation/       # Phase 4 question sets (dev, held-out v1-v5) and results
│   └── logs/             # answers.jsonl (every answer, for dashboards)
├── config/kora_config.json     # company configuration (thresholds, word list, access, programs)
├── reports/                    # evaluation reports + phase4_m5_findings.md
├── src/
│   ├── ingest.py               # Phase 2 entry point
│   ├── build_intelligence.py   # Phase 3 entry point
│   ├── explain.py              # explore the graph from the terminal
│   ├── employee.py             # one employee's context from the terminal
│   ├── ask.py                  # Phase 4: ask KORA (--why, --as, interactive)
│   ├── index.py                # Phase 4: index check / rebuild, answer logging
│   ├── retrieval/              # Phase 4: config, router, search, answer, access, embeddings (optional)
│   ├── evaluation/             # Phase 4: run_eval, calibrate_semantic
│   ├── pipeline.py, ingestion/, extraction/, cleaning/, chunking/, metadata/, validation/, storage/
│   └── intelligence/           # Phase 3 modules (one file per concern)
├── test/                       # pytest: Phase 2-4 (329 tests)
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

Phase 5: a UI and dashboards on top of `Kora.ask(question, user=...)`: ask with login (access control), show sources, the "why" and timings, edit the configuration (word list, access rules), and chart `data/logs/answers.jsonl` (refusals and their unknown words, latency, statuses). Later: a service layer / API, and an optional language-model step to rephrase questions and answers.
