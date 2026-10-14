"""
Phase 5 - the knowledge graph at RUNTIME.

Phase 3 builds the graph and saves it (data/intelligence/entities.json and
relationships.json). This package reads those saved files and adds what an
answer or a screen needs before it may use a relationship:

  load.py      saved files -> the Phase 3 KnowledgeGraph (no rebuild, ~0.1 s)
  schema.py    which (entity -> relationship -> entity) shapes are valid
  trace.py     every piece of evidence checked against its source again:
               a quoted sentence in its chunk, a header line in its file, a
               value in its CSV row. Result: a provenance TIER per link.
  view.py      what ONE user may see: hidden documents, their processes and
               topics, and colleagues' private details disappear -- links,
               names and counts included.
  quality.py   graph health report (python -m src.graph.quality)
"""
