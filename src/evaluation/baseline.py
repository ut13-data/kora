"""
Phase 5 M0 - freeze the Phase 4 results, so every later change is measured
against them instead of against memory.

    python -m src.evaluation.baseline --freeze      # once: writes data/evaluation/baseline_phase4.json
    python -m src.evaluation.baseline               # any time later: compare today's KORA with it

The baseline stores, for every question in every set, the status KORA gave
and the documents it cited. The comparison prints, per set, the numbers
then and now, and lists every question whose status changed:

    FIXED      wrong status before, right now
    BROKEN     right before, wrong now          <- must be explained before closing a milestone
    CHANGED    still right (or still wrong), but a different status or citations

The freeze refuses to overwrite an existing baseline (use --force), so the
"before" can't silently move.
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.run_eval import TOP_K, build, evaluate_question, load_dataset, summarise

BASELINE = Path("data/evaluation/baseline_phase4.json")
EVAL_DIR = Path("data/evaluation")

# name -> question file. The graph sets (Phase 5) are added here once they exist.
SETS = {
    "development": "eval_questions.json",
    "heldout_v1": "heldout_questions.json",
    "heldout_v2": "heldout_v2_questions.json",
    "heldout_v3": "heldout_v3_questions.json",
    "heldout_v4": "heldout_v4_questions.json",
    "heldout_v5": "heldout_v5_questions.json",
    "graph_dev": "graph_dev_questions.json",          # Phase 5: relationship questions (used while building)
    "graph_heldout": "graph_heldout_questions.json",  # Phase 5: run once, after M3
}
METRICS = ("status_accuracy", "key_fact_found", "names_found", "no_leak", "citation_precision",
           "faithfulness", "hit@5")


def available_sets(eval_dir: Path = EVAL_DIR, sets: dict = SETS) -> dict[str, Path]:
    return {name: eval_dir / file for name, file in sets.items() if (eval_dir / file).exists()}


def snapshot(kora, sets: dict[str, Path], top_k: int = TOP_K) -> dict:
    """Run every set once with the SAME loaded KORA; keep what a comparison needs."""
    out = {}
    for name, path in sets.items():
        rows = [evaluate_question(q, kora, top_k) for q in load_dataset(path)]
        kora_summary = summarise(rows)["all"]["kora"]
        out[name] = {
            "file": path.name,
            "questions": len(rows),
            "summary": {m: kora_summary.get(m) for m in METRICS},
            "per_question": {r["id"]: {"question": r["question"], "status": r["kora"]["status"],
                                       "status_correct": r["kora"]["status_correct"],
                                       "cited": sorted(set(r["kora"]["cited"])),
                                       "faithful": r["kora"]["faithful"],
                                       "graph": r["kora"].get("graph"),
                                       "names_found": r["kora"]["names_found"],
                                       "leaks": r["kora"]["leaks"]}
                             for r in rows},
        }
    return out


def compare(before: dict, now: dict) -> dict:
    """Plain differences; no judgement beyond FIXED / BROKEN / CHANGED."""
    report = {"sets": {}, "fixed": [], "broken": [], "changed": [], "missing": [], "new": []}
    for name in sorted(set(before) | set(now)):
        b, n = before.get(name), now.get(name)
        if b is None or n is None:
            (report["new"] if b is None else report["missing"]).append(name)
            continue
        report["sets"][name] = {m: (b["summary"].get(m), n["summary"].get(m)) for m in METRICS}
        for qid, old in b["per_question"].items():
            new = n["per_question"].get(qid)
            if new is None:
                report["missing"].append(f"{name}/{qid}")
                continue
            item = {"set": name, "id": qid, "question": old["question"],
                    "before": old["status"], "now": new["status"],
                    "cited_before": old["cited"], "cited_now": new["cited"]}
            if new.get("leaks"):                     # a leak is never a fix, whatever the status
                report["broken"].append({**item, "now": f"{new['status']} LEAKS {new['leaks']}"})
            elif not old["status_correct"] and new["status_correct"]:
                report["fixed"].append(item)
            elif old["status_correct"] and not new["status_correct"]:
                report["broken"].append(item)
            elif old["status"] != new["status"] or old["cited"] != new["cited"]:
                report["changed"].append(item)
        report["new"] += [f"{name}/{q}" for q in n["per_question"] if q not in b["per_question"]]
    return report


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def render(report: dict) -> str:
    short = {"status_accuracy": "status", "key_fact_found": "key fact", "names_found": "names",
             "no_leak": "no leak", "citation_precision": "cite prec", "faithfulness": "faithful", "hit@5": "hit@5"}
    lines = ["Before -> now (+ better, - worse)", "",
             "Set           " + " ".join(f"{short[m]:>14}" for m in METRICS)]
    for name, metrics in report["sets"].items():
        cells = []
        for m in METRICS:
            old, new = metrics[m]
            mark = "" if old == new else (" +" if (new or 0) > (old or 0) else " -")
            cells.append(f"{_fmt(old) + '->' + _fmt(new) + mark:>14}")
        lines.append(f"{name:<13} " + " ".join(cells))
    for title, key in (("BROKEN (right before, wrong now)", "broken"), ("FIXED", "fixed"),
                       ("CHANGED (same correctness)", "changed")):
        lines += ["", f"{title}: {len(report[key])}"]
        for i in report[key]:
            cites = "" if i["cited_before"] == i["cited_now"] else \
                f"  cited {i['cited_before']} -> {i['cited_now']}"
            lines.append(f"  {i['set']}/{i['id']} {i['before']} -> {i['now']}{cites}  | {i['question']}")
    if report["missing"] or report["new"]:
        lines += ["", f"Missing now: {report['missing'] or 'none'}", f"New since baseline: {report['new'] or 'none'}"]
    return "\n".join(lines)


def report_markdown(text: str, report: dict) -> str:
    """The comparison as a Markdown page: a short reading guide, then the exact table."""
    verdict = "No question got worse." if not report["broken"] else \
        f"{len(report['broken'])} question(s) got worse. See BROKEN below."
    return "\n".join([
        "# Phase 5 compared with Phase 4", "",
        "Made by `python -m src.evaluation.baseline --report reports/phase5_final_evaluation.md`.", "",
        f"**{verdict}** FIXED: {len(report['fixed'])}. CHANGED with the same correctness: {len(report['changed'])}.", "",
        "How to read it:", "",
        "* *before* is the Phase 4 result frozen at the start of Phase 5 (`data/evaluation/baseline_phase4.json`).",
        "* *names* checks that the people, roles or teams the question is about appear in the answer.",
        "* *no leak* checks that no hidden document is quoted, cited or named for that employee. A new leak always counts as BROKEN.",
        "* *cite prec* leaves out company-record citations (CSV rows), which have no document chunk.",
        "* `cited [...] -> []` means the answer now cites a document header or a company record instead of a passage.",
        "* The graph sets were written during Phase 5 by the builder; held-out H02, H04 and H09 were seen during building.", "",
        "```", text, "```", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or compare against the Phase 4 baseline.")
    parser.add_argument("--freeze", action="store_true", help="write the baseline (once)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing baseline")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--report", type=Path, help="also save the comparison as a Markdown file")
    args = parser.parse_args()

    sets = available_sets()
    setup = build()
    now = snapshot(setup.kora, sets)

    if args.freeze:
        if args.baseline.exists() and not args.force:
            print(f"{args.baseline} already exists; it is the 'before'. Use --force only if you mean to move it.")
            return 1
        payload = {"description": "Phase 4 closing results, frozen at the start of Phase 5 (M0). "
                                  "Every Phase 5 change is compared with this file.",
                   "semantic": setup.kora.retriever.semantic_status, "top_k": TOP_K, "sets": now}
        args.baseline.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        for name, s in now.items():
            print(f"{name:<13} n={s['questions']:<3} " +
                  "  ".join(f"{m} {_fmt(s['summary'][m])}" for m in METRICS))
        print(f"Wrote {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}. Run with --freeze first.")
        return 1
    before = json.loads(args.baseline.read_text(encoding="utf-8"))["sets"]
    report = compare(before, now)
    text = render(report)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_markdown(text, report), encoding="utf-8")
        print(f"\nWrote {args.report}")
    return 1 if report["broken"] else 0


if __name__ == "__main__":
    sys.exit(main())
