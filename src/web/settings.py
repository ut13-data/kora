"""
Phase 5 M5 - editing the company config from the admin screen, safely.

    check(raw, employees, admin)   -> list of problems (empty = valid)
    save(path, raw)                -> backup name; the old file is kept in config/backups/
    backups(path) / restore(path, name)

Rules:
  * The SAME validator KORA uses at start-up (parse_config) decides; a config
    with any problem is never written, and every problem is listed at once.
  * Extra checks only the UI needs: admin IDs and emp:<id> access groups must
    be real employees, and an admin can't save a config that removes their own
    admin rights or leaves no admin (nobody could fix the settings afterwards).
  * Every save keeps the previous file as a timestamped backup; restoring a
    backup is itself a save (so the restore can be undone too).
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.config import THRESHOLDS, ConfigError, parse_config

BACKUP_DIR = "backups"
_BACKUP_NAME = re.compile(r"^kora_config\.\d{8}T\d{6}\d*Z\.json$")


def check(raw, employee_ids: set[str], admin_id: str | None) -> list[str]:
    try:
        cfg = parse_config(raw, "the new settings")
    except ConfigError as e:
        return [line.strip()[2:] if line.strip().startswith("- ") else line.strip()
                for line in str(e).splitlines()[1:] if line.strip()]
    problems = []
    unknown = [a for a in cfg.admins if a not in employee_ids]
    if unknown:
        problems.append(f"admins: unknown employee ID(s) {unknown}")
    if not cfg.admins:
        problems.append("admins: keep at least one admin, or nobody can change the settings again")
    elif admin_id and admin_id not in cfg.admins:
        problems.append(f"admins: you ({admin_id}) would lose admin access; add another admin and let them remove you")
    for pattern, groups in cfg.access_rules.items():
        for g in groups:
            if g.startswith("emp:") and g[4:] not in employee_ids:
                problems.append(f"access.rules.{pattern}: unknown employee in group '{g}'")
    return problems


def schema(employee_columns: list[str] | None = None) -> dict:
    """What the admin screen needs to build its forms."""
    return {"thresholds": {k: {"default": d, "min": lo, "max": hi, "whole_number": isinstance(d, int)}
                           for k, (d, lo, hi) in THRESHOLDS.items()},
            "groups_help": "all | manager | dept:<id> | loc:<id> | role:<id> | emp:<id>",
            "employee_columns": [c for c in (employee_columns or [])
                                 if c not in ("employee_id", "first_name", "last_name")]}


def _backup_dir(path: Path) -> Path:
    return Path(path).parent / BACKUP_DIR


def save(path: Path, raw: dict) -> str | None:
    """Write RAW (already checked). Returns the backup name of the previous file, if there was one."""
    path = Path(path)
    name = None
    if path.exists():
        d = _backup_dir(path)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"kora_config.{stamp}.json"
        shutil.copy2(path, d / name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)                       # never a half-written config
    return name


def backups(path: Path) -> list[dict]:
    d = _backup_dir(path)
    files = sorted((p for p in d.glob("kora_config.*.json") if _BACKUP_NAME.match(p.name)), reverse=True) \
        if d.exists() else []
    return [{"name": p.name, "saved_at": p.name.split(".")[1], "bytes": p.stat().st_size} for p in files]


def read_backup(path: Path, name: str) -> dict:
    if not _BACKUP_NAME.match(name):              # no paths, no "..": only names this module wrote
        raise FileNotFoundError(name)
    p = _backup_dir(path) / name
    if not p.exists():
        raise FileNotFoundError(name)
    return json.loads(p.read_text(encoding="utf-8"))
