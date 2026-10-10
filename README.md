# KORA: Knowledge Object Retrieval Agent

KORA turns a company's internal knowledge (policies, processes, product docs, org data) into a clean, structured, chunked dataset with rich metadata and relationships, ready for retrieval and an AI assistant in later phases.

The source data is **Nexora Systems**, a synthetic mid-sized B2B software company (53 Markdown documents, 7 CSV tables).

## Status

**Phase 2 (Ingestion Pipeline): complete.**

Out of scope for this phase, planned for later: embeddings, vector store, retrieval, LLM calls, agent, and the SkillKitLearn frontend.

## Pipeline

```
Upload -> Extract -> Clean -> Identify metadata -> Chunk -> Relationships -> Validate -> Store
```

| Stage | What it does |
|---|---|
| Upload / Input | Identifies file name, type, size, source path, ingestion timestamp. Rejects unsupported types. |
| Extraction | Markdown, TXT, PDF, DOCX to plain text. CSV stays tabular (one dict per row). |
| Cleaning | Removes whitespace noise and encoding artifacts. Never changes words or numbers. |
| Document identification | Reads the `**Label:** value` header (ID, title, department, version, status, related processes/roles). Missing fields are `unknown`, never invented. |
| Chunking | Structure-aware: splits Markdown at headings, then by size (about 1,200 characters) on paragraph, line or word boundaries. Each chunk keeps its `section` and a `parent_id` for its full section. Stable chunk IDs such as `PROC-HR-001_v3.0_002`. |
| Entity registry | One lookup of every department, role, process, product, location, employee and document from the CSVs. |
| Relationships | Built from document headers, CSV tables, text discovery (mentions of known entities and glossary topics) and location links. Every relationship records its evidence. |
| Validation | Reports problems, never silently fixes them: missing or duplicate IDs, version histories, header vs master-index mismatches, broken relationship ends, duplicate chunks, employees without a manager, skipped files. |
| Store | Writes JSON outputs (see below). |

## Quick start

Requires Python 3.10+ (developed on 3.14).

```
python -m venv venvkora
venvkora\Scripts\activate
pip install -r requirements.txt
```

Put the knowledge files in `data/raw/` (subfolders are scanned), then:

```
python -m src.ingest                                  # full run, archived documents included
python -m src.ingest --exclude-archived --no-save     # current documents only, nothing written
python -m pytest -v                                   # run the test suite
```

From Python (this is the interface a frontend should call):

```python
from src.ingest import IngestionSettings, run_ingestion

summary, result = run_ingestion(IngestionSettings(include_archived=True))
print(summary.to_dict())   # JSON-friendly counts, issues, files written
```

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `raw_dir` | `data/raw` | Where the knowledge files are |
| `include_archived` | `True` | Keep archived documents so employees can see what a rule was before and what it is now |
| `chunk_size` | `1200` | Maximum characters per chunk |
| `chunk_overlap` | `100` | Overlap, used only when one section is split into several chunks |
| `save_output` | `True` | Write output files |

## Output

```
data/processed/documents.json          one metadata record per document
data/processed/relationships.json      every relationship, with evidence
data/processed/tables/<name>.json      one file per CSV, rows unchanged
data/processed/validation_report.json  summary, skipped files, unresolved references, issues
data/chunks/chunks.jsonl               one chunk per line (JSON Lines)
```

### Relationship types

| Source | Types |
|---|---|
| Document headers | `document_related_to_process`, `document_related_to_document`, `document_related_to_role`, `document_for_audience` |
| CSV tables | `employee_has_role`, `employee_at_location`, `employee_reports_to`, `role_in_department`, `role_related_to_process`, `process_owned_by_department`, `process_documented_in`, `product_owned_by_department`, `product_documented_in` |
| Text discovery | `document_mentions_process`, `document_mentions_document`, `document_mentions_product`, `document_mentions_role`, `document_mentions_location`, `document_has_topic`, `process_related_to_product` |
| Location links | `location_has_checklist`, `culture_tone_guidance` |

## Project structure

```
KORA/
├── data/
│   ├── raw/          # source files (never modified)
│   ├── processed/    # JSON output
│   └── chunks/       # chunks.jsonl
├── src/
│   ├── ingest.py           # entry point: run_ingestion() and the command line
│   ├── pipeline.py         # runs every stage in order
│   ├── ingestion/          # file intake
│   ├── extraction/         # text and table extraction
│   ├── cleaning/           # noise removal
│   ├── chunking/           # structure-aware chunking
│   ├── metadata/           # header metadata, registry, relationships,
│   │                       # discovery, location links
│   ├── validation/         # validation checks
│   └── storage/            # JSON output
├── tests/                  # pytest: unit tests and Nexora integration tests
├── requirements.txt
└── README.md
```

## Design decisions

- **Resolve or report, never guess.** Header values are resolved to real registry IDs, recognised as placeholders (`Multiple`, `N/A`), modelled as audiences, or reported as unresolved.
- **Aliases only from Nexora's own documents** (for example `Project Lead` = `Implementation Manager`, per the glossary).
- **Archived documents are included by default** and tagged on every chunk.
- **Audiences** (`All Employees`, `International New Hires (Germany)`) are modelled as their own relationship type, not forced into roles.
- **Departments are not matched in free text**, because names like "Sales" or "Operations" are everyday words.
- **Discovery is keyword matching, not understanding.** Every discovered link stores which chunk it came from, so it can be checked.
- **Chunk size is a reasoned starting point**, verified by answer-containment tests. The real test, retrieval quality, comes in the next phase.

## Known data findings (reported by the validator)

These are issues in the Nexora source data, not in KORA:

1. `PROC-IMPL-001` exists as v2.0 (Active) and v1.0 (Archived). Kept on purpose as version history.
2. The archived onboarding file's header says `PROC-IMPL-001`, but `documents.csv` lists it as `PROC-IMPL-001-V1`.
3. 14 employees have no `manager_id` in `employees.csv`.

## Next phase

Embeddings and retrieval over `chunks.jsonl`, using `section` for contextual chunk headers and `parent_id` for small-to-big retrieval, then the SkillKitLearn frontend calling `run_ingestion()`.
