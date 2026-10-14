"""
Phase 4 M5 - evaluate retrieval and answers, against a baseline.

    python -m src.evaluation.run_eval                  # writes results + report
    python -m src.evaluation.run_eval --no-save        # print the summary only

Two systems answer the same questions (data/evaluation/eval_questions.json):

  baseline   plain TF-IDF over every chunk (M1 tier 3 only): no router, no
             named-document tier, archived chunks allowed.
  kora       router + tiers (M2) feeding the answer layer (M3 + M4): the
             ASRS-style "narrow first" design.

Retrieval is compared on both. Answer quality (status, key fact, citations,
faithfulness) exists only for KORA, since the baseline has no answer layer.

Gold labels are document IDs, so they survive re-chunking. Every metric is
plain arithmetic over those labels -- nothing is judged by a model.

Outputs:
  data/evaluation/results.json      every question, every metric (Phase 5 can chart this)
  reports/evaluation_results.md     tables generated from results.json
"""

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from src.retrieval.answer import KoraAnswerer
from src.retrieval.router import normalise

DATASET = Path("data/evaluation/eval_questions.json")
RESULTS = Path("data/evaluation/results.json")
REPORT = Path("reports/evaluation_results.md")
TOP_K = 5


# ---------------- dataset ----------------

def load_dataset(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def check_labels(questions: list[dict], chunks) -> list[str]:
    """Problems with the LABELS themselves (not with KORA): unknown document
    IDs, or an expected fact that no gold/acceptable document contains.
    Run before scoring, so a 'failure' can't be a labelling mistake."""
    ids = {c.document_id for c in chunks}
    text: dict[str, list[str]] = {}
    for c in chunks:
        text.setdefault(c.document_id, []).append(c.chunk_text)
    norm = {k: normalise(" ".join(v)) for k, v in text.items()}
    problems = []
    for q in questions:
        docs = q["gold_documents"] + q["acceptable_documents"]
        for d in docs:
            if d not in ids:
                problems.append(f"{q['id']}: unknown document {d}")
        if q.get("expected_contains"):
            key = normalise(q["expected_contains"])
            if not any(key in norm.get(d, "") for d in docs):
                problems.append(f"{q['id']}: expected fact not in any gold/acceptable document")
    return problems


# ---------------- metrics (plain arithmetic) ----------------

def retrieval_metrics(retrieved_docs: list[str], gold: list[str]) -> dict | None:
    """hit@k, recall@k and reciprocal rank for one ranked list of document IDs
    (one entry per retrieved chunk, so a document can repeat). None when the
    question has no gold documents (no-answer questions)."""
    if not gold:
        return None
    gold_set = set(gold)
    found = gold_set & set(retrieved_docs)
    first = next((i for i, d in enumerate(retrieved_docs, 1) if d in gold_set), None)
    return {"hit": bool(found), "recall": len(found) / len(gold_set),
            "reciprocal_rank": 1 / first if first else 0.0}


def citation_precision(cited_docs: list[str], gold: list[str], acceptable: list[str]) -> float | None:
    """Share of cited documents that are gold or acceptable. None if nothing cited."""
    cited = set(cited_docs)
    if not cited:
        return None
    return len(cited & (set(gold) | set(acceptable))) / len(cited)


def contains_fact(answer, fact: str | None) -> bool | None:
    """Does the answer's text (lines + their details) contain the key fact?"""
    if not fact:
        return None
    text = normalise(" ".join(f"{l.text} {l.detail}" for l in answer.lines))
    return normalise(fact) in text


def _answer_text(answer) -> str:
    return " ".join(f"{l.text} {l.detail}" for l in answer.lines).lower()


def names_found(answer, names: list[str] | None) -> float | None:
    """Phase 5: share of expected names (roles, departments, processes) the
    answer states. Case-insensitive substring; None if the question lists none."""
    if not names:
        return None
    text = _answer_text(answer)
    return sum(n.lower() in text for n in names) / len(names)


def leaks(answer, forbidden: list[str] | None) -> list[str]:
    """Phase 5: forbidden names that appear ANYWHERE the user could see them:
    answer lines, notes, citation titles, document IDs and excerpts.

    One exception: a name the user typed in the question may be echoed back in
    a note ("the knowledge base never mentions zimmermann") -- that repeats the
    user's own words and reveals nothing. It still counts in lines and citations."""
    if not forbidden:
        return []
    found_data = " ".join([_answer_text(answer)] +
                          [f"{c.title} {c.document_id} {c.excerpt}" for c in answer.citations]).lower()
    notes = " ".join(answer.notes).lower()
    asked = answer.question.lower()
    return [f for f in forbidden
            if f.lower() in found_data or (f.lower() in notes and f.lower() not in asked)]


def faithful(answer, kora: KoraAnswerer) -> bool:
    """Every answer line passes KORA's own runtime verifier (one definition of
    'faithful', so the evaluation and the product can never disagree), and
    every line's citation number points at the chunk the line came from."""
    by_number = {c.number: c.chunk_id for c in answer.citations}
    return all(kora.verify_line(l) and (l.citation is None or by_number.get(l.citation) == l.chunk_id)
               for l in answer.lines)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _pct(values, p):
    values = sorted(values)
    if not values:
        return None
    k = max(0, min(len(values) - 1, round(p / 100 * (len(values) - 1))))
    return round(values[k], 2)


# ---------------- run ----------------

@dataclass
class Setup:
    kora: KoraAnswerer
    load_seconds: float
    peak_memory_mb: float
    index_build_seconds: dict


def build(**dirs) -> Setup:
    """Load everything once, measuring time and peak Python memory. Index
    build times are measured separately for the baseline (keyword + TF-IDF)
    and for what KORA adds on top (router catalogue)."""
    from src.intelligence.loader import load_phase2
    from src.retrieval.router import Router
    from src.retrieval.search import build_retriever

    # Time first, WITHOUT the memory tracker: tracemalloc slows every
    # allocation it watches (it made a 8 s start-up look like 38 s). Memory is
    # then measured on a second, separate load.
    t0 = time.perf_counter()
    kora = KoraAnswerer.load(**dirs)
    load_seconds = time.perf_counter() - t0
    tracemalloc.start()
    KoraAnswerer.load(**dirs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    phase2 = load_phase2(*(dirs.get(k, Path(v)) for k, v in
                           (("processed_dir", "data/processed"), ("chunks_dir", "data/chunks"),
                            ("raw_dir", "data/raw"))))
    t0 = time.perf_counter()
    build_retriever(phase2)
    base_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    Router.from_files(dirs.get("intelligence_dir", Path("data/intelligence")))
    router_build = time.perf_counter() - t0
    return Setup(kora, round(load_seconds, 3), round(peak / 1e6, 1),
                 {"baseline_indexes": round(base_build, 4), "kora_router_extra": round(router_build, 4)})


def evaluate_question(q: dict, kora: KoraAnswerer, top_k: int = TOP_K) -> dict:
    chunks = kora.retriever.chunks

    t0 = time.perf_counter()
    base = chunks.search(q["question"], top_k=top_k, use_semantic=False)   # plain TF-IDF, words only
    base_ms = (time.perf_counter() - t0) * 1000

    user = None
    if q.get("as_user"):                          # Phase 5: ask as an employee (access control)
        user = kora.retriever.users.get(q["as_user"])
        if user is None:
            raise ValueError(f"{q['id']}: unknown as_user {q['as_user']}")
    answer = kora.answer(q["question"], top_k=top_k, user=user)
    kora_docs = [r.chunk.document_id for r in answer.retrieval.results]
    base_docs = [r.chunk.document_id for r in base]
    cited = [c.document_id for c in answer.citations] if answer.status != "not_found" else []

    return {
        "id": q["id"], "category": q["category"], "question": q["question"],
        "seen_in_development": q["seen_in_development"],
        "gold_documents": q["gold_documents"],
        "baseline": {"retrieved": base_docs, "metrics": retrieval_metrics(base_docs, q["gold_documents"]),
                     "ms": round(base_ms, 2)},
        "kora": {
            "retrieved": kora_docs,
            "metrics": retrieval_metrics(kora_docs, q["gold_documents"]),
            "retrieval_ms": answer.retrieval.elapsed_ms,
            "answer_ms": round(answer.elapsed_ms, 2),
            "status": answer.status,
            "status_correct": answer.status in q["expected_status"],
            "expected_status": q["expected_status"],
            "contains_fact": contains_fact(answer, q.get("expected_contains")),
            "citation_precision": citation_precision(cited, q["gold_documents"], q["acceptable_documents"]),
            "faithful": faithful(answer, kora),
            "names_found": names_found(answer, q.get("expected_names")),
            "leaks": leaks(answer, q.get("forbidden_names")),
            "cited": cited,
            "intents": answer.retrieval.plan.intents,
            "reasons": answer.retrieval.plan.reasons,
            "notes": answer.notes,
        },
    }


def summarise(rows: list[dict]) -> dict:
    def block(subset):
        def m(system, key):
            return _mean([r[system]["metrics"][key] for r in subset if r[system]["metrics"]])
        return {
            "questions": len(subset),
            "baseline": {"hit@5": m("baseline", "hit"), "recall@5": m("baseline", "recall"),
                         "mrr": m("baseline", "reciprocal_rank")},
            "kora": {"hit@5": m("kora", "hit"), "recall@5": m("kora", "recall"),
                     "mrr": m("kora", "reciprocal_rank"),
                     "status_accuracy": _mean([r["kora"]["status_correct"] for r in subset]),
                     "key_fact_found": _mean([r["kora"]["contains_fact"] for r in subset]),
                     "citation_precision": _mean([r["kora"]["citation_precision"] for r in subset]),
                     "faithfulness": _mean([r["kora"]["faithful"] for r in subset]),
                     "names_found": _mean([r["kora"].get("names_found") for r in subset]),
                     "no_leak": _mean([not r["kora"].get("leaks") for r in subset])},
        }

    categories = sorted({r["category"] for r in rows})
    base_ms = [r["baseline"]["ms"] for r in rows]
    kora_ms = [r["kora"]["answer_ms"] for r in rows]
    return {
        "all": block(rows),
        "new_questions_only": block([r for r in rows if not r["seen_in_development"]]),
        "seen_in_development": block([r for r in rows if r["seen_in_development"]]),
        "by_category": {c: block([r for r in rows if r["category"] == c]) for c in categories},
        "latency_ms": {"baseline_retrieval_p50": _pct(base_ms, 50), "baseline_retrieval_p95": _pct(base_ms, 95),
                       "kora_answer_p50": _pct(kora_ms, 50), "kora_answer_p95": _pct(kora_ms, 95)},
    }


def run(top_k: int = TOP_K, dataset: Path = DATASET, **dirs) -> dict:
    questions = load_dataset(dataset)
    setup = build(**dirs)
    problems = check_labels(questions, list(setup.kora.retriever.chunks.keyword.chunks.values()))
    rows = [evaluate_question(q, setup.kora, top_k) for q in questions]
    return {
        "dataset": str(dataset),
        "semantic": setup.kora.retriever.semantic_status,
        "top_k": top_k,
        "label_problems": problems,
        "resources": {"load_seconds": setup.load_seconds, "startup_steps_s": setup.kora.startup_summary(),
                      "peak_python_memory_mb": setup.peak_memory_mb,
                      "index_build_seconds": setup.index_build_seconds,
                      "chunks": len(setup.kora.retriever.chunks.keyword.chunks)},
        "summary": summarise(rows),
        "questions": rows,
    }


# ---------------- report ----------------

def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def build_report(results: dict) -> str:
    s = results["summary"]
    out = ["# KORA Phase 4 - Evaluation results", "", f"Dataset: `{results.get('dataset')}`", "",
           "_Generated by `python -m src.evaluation.run_eval` from `data/evaluation/results.json`. "
           "Do not edit by hand._", "",
           f"Embeddings: {results.get('semantic', 'n/a')}", "",
           f"{len(results['questions'])} questions, top-{results['top_k']} retrieval. "
           f"Label problems: {len(results['label_problems'])}.", "",
           "The same author (Claude) built KORA and wrote these questions. Questions flagged "
           "`seen_in_development` were used while building M1-M4; the **new questions** row is the fairer number.", ""]

    out += ["## Retrieval: baseline vs KORA", "",
            "| Set | n | Baseline hit@5 | KORA hit@5 | Baseline recall@5 | KORA recall@5 | Baseline MRR | KORA MRR |",
            "|---|---|---|---|---|---|---|---|"]
    for name, key in (("All", "all"), ("New questions only", "new_questions_only"),
                      ("Seen in development", "seen_in_development")):
        b = s[key]
        out.append(f"| {name} | {b['questions']} | {_fmt(b['baseline']['hit@5'])} | {_fmt(b['kora']['hit@5'])} | "
                   f"{_fmt(b['baseline']['recall@5'])} | {_fmt(b['kora']['recall@5'])} | "
                   f"{_fmt(b['baseline']['mrr'])} | {_fmt(b['kora']['mrr'])} |")

    out += ["", "## Answer quality (KORA only)", "",
            "| Set | Status accuracy | Key fact found | Citation precision | Faithfulness |", "|---|---|---|---|---|"]
    for name, key in (("All", "all"), ("New questions only", "new_questions_only"),
                      ("Seen in development", "seen_in_development")):
        k = s[key]["kora"]
        out.append(f"| {name} | {_fmt(k['status_accuracy'])} | {_fmt(k['key_fact_found'])} | "
                   f"{_fmt(k['citation_precision'])} | {_fmt(k['faithfulness'])} |")

    out += ["", "## By category", "",
            "| Category | n | Baseline hit@5 | KORA hit@5 | KORA recall@5 | Status acc. | Key fact |",
            "|---|---|---|---|---|---|---|"]
    for cat, b in s["by_category"].items():
        out.append(f"| {cat} | {b['questions']} | {_fmt(b['baseline']['hit@5'])} | {_fmt(b['kora']['hit@5'])} | "
                   f"{_fmt(b['kora']['recall@5'])} | {_fmt(b['kora']['status_accuracy'])} | "
                   f"{_fmt(b['kora']['key_fact_found'])} |")

    r = results["resources"]
    lat = s["latency_ms"]
    out += ["", "## Latency and resources", "",
            f"- Baseline retrieval: p50 {_fmt(lat['baseline_retrieval_p50'])} ms, p95 {_fmt(lat['baseline_retrieval_p95'])} ms",
            f"- KORA full answer (route + retrieve + answer): p50 {_fmt(lat['kora_answer_p50'])} ms, "
            f"p95 {_fmt(lat['kora_answer_p95'])} ms",
            f"- Load time (Phase 2 + 3 files, indexes, embeddings if on): {r['load_seconds']} s; peak Python heap "
            f"{r['peak_python_memory_mb']} MB (tracemalloc: Python objects only, not model weights)",
            "- Start-up by step (s): " + ", ".join(f"{k} {v}" for k, v in r.get("startup_steps_s", {}).items()),
            f"- Index build: baseline {r['index_build_seconds']['baseline_indexes']} s, "
            f"KORA router adds {r['index_build_seconds']['kora_router_extra']} s ({r['chunks']} chunks)", ""]

    out += ["## Failures (KORA)", ""]
    any_fail = False
    for q in results["questions"]:
        k = q["kora"]
        problems = []
        if k["metrics"] and not k["metrics"]["hit"]:
            problems.append("missed every gold document")
        if not k["status_correct"]:
            problems.append(f"status {k['status']} (expected {'/'.join(k['expected_status'])})")
        if k["contains_fact"] is False:
            problems.append("key fact missing from answer")
        if k["citation_precision"] is not None and k["citation_precision"] < 1:
            problems.append(f"citation precision {k['citation_precision']:.2f} (cited {', '.join(k['cited'])})")
        if not k["faithful"]:
            problems.append("UNFAITHFUL line")
        if problems:
            any_fail = True
            seen = " (seen)" if q["seen_in_development"] else ""
            out.append(f"- **{q['id']}**{seen} [{q['category']}] {q['question']}  \n  " + "; ".join(problems))
    if not any_fail:
        out.append("None.")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KORA retrieval and answers against a baseline.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET,
                        help="Question set (default: the 41 development questions)")
    parser.add_argument("--name", default=None, help="Output suffix, e.g. 'heldout' -> results_heldout.json")
    args = parser.parse_args()

    results = run(args.top_k, args.dataset)
    global RESULTS, REPORT
    if args.name:
        RESULTS = RESULTS.with_name(f"results_{args.name}.json")
        REPORT = REPORT.with_name(f"evaluation_{args.name}.md")
    for p in results["label_problems"]:
        print(f"[LABEL] {p}")
    print(f"embeddings: {results['semantic']}")
    print(json.dumps({k: results["summary"][k] for k in ("all", "new_questions_only", "latency_ms")}, indent=2))
    print(json.dumps(results["resources"], indent=2))
    if not args.no_save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(build_report(results), encoding="utf-8")
        print(f"Wrote {RESULTS} and {REPORT}")


if __name__ == "__main__":
    main()
