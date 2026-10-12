"""Write Phase 3 outputs as readable, UTF-8 JSON (same conventions as Phase 2's store)."""

import json
from datetime import date, datetime
from pathlib import Path


def _to_jsonable(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Don't know how to store type {type(obj).__name__}")


def write_json(data, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_to_jsonable)
    return path
