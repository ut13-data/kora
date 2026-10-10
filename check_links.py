from pathlib import Path

from src.pipeline import process_folder
from src.metadata.relationships import RelationType

result = process_folder(Path("data/raw"))

print("Process -> Product links:")
for r in result.relationships:
    if r.relation_type == RelationType.PROCESS_RELATED_TO_PRODUCT:
        print(f"  {r.source} --> {r.target}   [{r.evidence}]")

print("\nprocesses.csv rows whose doc_id is PROC-IMPL-001:")
for proc_id, rows in result.registry.by_id["processes"].items():
    for row in rows:
        if row.get("doc_id") == "PROC-IMPL-001":
            print(f"  {proc_id}")