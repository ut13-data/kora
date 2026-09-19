# kora
Knowledge Object Retrieval Agent
# KORA — Knowledge Object Retrieval Agent

KORA is a Python project to build a knowledge ingestion pipeline for a synthetic company knowledge base (Nexora Systems). This repo currently covers **Phase 2: Ingestion Pipeline** only.

## Pipeline

```
Upload → Extract → Clean → Chunk → Metadata → Store
```

Out of scope for this phase: RAG, embeddings, vector database, LLM calls, agent logic, retrieval, and the final UI. Those come later.

## Progress

- [x] **Stage 1 — Upload/Input Layer**: identifies filename, file type, size, source path, and ingestion timestamp for supported files (Markdown, CSV, TXT, PDF, DOCX). Rejects unsupported file types with a clear error.
- [x] **Stage 2 — Extraction**: extracts text content from TXT, Markdown, DOCX, and CSV files into normalized objects (`ExtractedDocument` for text, `ExtractedTable` for CSV rows). PDF extraction implemented, not yet tested end-to-end.
- [ ] Stage 3 — Cleaning
- [ ] Stage 4 — Document Identification (metadata)
- [ ] Stage 5 — Chunking
- [ ] Stage 6 — Metadata Layer (entity/relationship design)
- [ ] Stage 7 — Validation

## Project structure

```
KORA/
├── data/
│   ├── raw/          # Original uploaded files, untouched
│   ├── processed/    # Cleaned/normalized documents (Stage 3+)
│   └── chunks/       # Chunked output (Stage 5+)
│
├── src/
│   ├── ingestion/     # Stage 1
│   ├── extraction/    # Stage 2
│   ├── cleaning/      # Stage 3
│   ├── chunking/      # Stage 5
│   ├── metadata/      # Stage 4 & 6
│   └── validation/    # Stage 7
│
├── tests/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- `python-docx`
- `pypdf`

## Notes

Built for eventual integration with a SkillKitLearn frontend. Ingestion functionality is exposed as plain Python functions/classes with predictable inputs and outputs for that purpose — no frontend work has been done yet.
