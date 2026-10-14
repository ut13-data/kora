"""Phase 4 M5: the evaluation's own arithmetic and labels must be right."""

import pytest

from src.evaluation.run_eval import (
    check_labels, citation_precision, load_dataset, retrieval_metrics, summarise,
)


def test_retrieval_metrics_hit_recall_mrr():
    m = retrieval_metrics(["A", "B", "C", "B"], ["B", "D"])
    assert m == {"hit": True, "recall": 0.5, "reciprocal_rank": 0.5}


def test_retrieval_metrics_miss():
    assert retrieval_metrics(["A", "C"], ["B"]) == {"hit": False, "recall": 0.0, "reciprocal_rank": 0.0}


def test_retrieval_metrics_skipped_without_gold():
    assert retrieval_metrics(["A"], []) is None


def test_citation_precision_counts_acceptable_documents():
    assert citation_precision(["A", "B", "C"], gold=["A"], acceptable=["B"]) == pytest.approx(2 / 3)
    assert citation_precision([], gold=["A"], acceptable=[]) is None


def test_dataset_covers_every_brief_category():
    categories = {q["category"] for q in load_dataset()}
    assert categories >= {"exact_fact", "synonym", "multi_document", "role_process", "version",
                          "conflict", "missing_info", "outdated", "germany", "communication", "no_answer"}


def test_dataset_ids_unique_and_statuses_valid():
    questions = load_dataset()
    assert len({q["id"] for q in questions}) == len(questions)
    valid = {"answered", "partial", "conflict", "not_found"}
    assert all(set(q["expected_status"]) <= valid for q in questions)


def test_labels_match_the_real_documents(intel):
    """Every gold document exists and every expected fact is really in a
    gold/acceptable document -- so a failure is KORA's, not a labelling slip."""
    assert check_labels(load_dataset(), intel.phase2.chunks) == []


def test_summary_separates_new_and_seen_questions():
    row = {"category": "x", "seen_in_development": False,
           "baseline": {"metrics": {"hit": True, "recall": 1.0, "reciprocal_rank": 1.0}, "ms": 1.0},
           "kora": {"metrics": {"hit": False, "recall": 0.0, "reciprocal_rank": 0.0}, "answer_ms": 2.0,
                    "status_correct": True, "contains_fact": None, "citation_precision": None, "faithful": True}}
    seen = {**row, "seen_in_development": True}
    s = summarise([row, seen])
    assert s["new_questions_only"]["questions"] == 1
    assert s["seen_in_development"]["questions"] == 1
    assert s["all"]["baseline"]["hit@5"] == 1.0 and s["all"]["kora"]["hit@5"] == 0.0
