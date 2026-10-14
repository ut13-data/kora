"""Phase 4 M3 + M4: grounded answers, citations, conflicts, gaps, abstention."""

import json

import pytest

from src.retrieval.answer import KoraAnswerer, split_sentences, time_window
from src.retrieval.router import normalise


@pytest.fixture(scope="module")
def kora(intel):
    return KoraAnswerer.from_intelligence(intel)


def _docs(answer):
    return {c.document_id for c in answer.citations}


# ---------------- helpers ----------------

def test_split_sentences_keeps_paragraph_under_heading():
    """Bug found in M3: the paragraph right under a heading was dropped."""
    text = "## Parental Leave\nLocation-specific statutory entitlements apply as a floor; HR can\nprovide detail."
    assert split_sentences(text) == [
        "Location-specific statutory entitlements apply as a floor; HR can provide detail."]


def test_split_sentences_keeps_faq_question_with_its_answer():
    text = "# FAQ\n**Q: Who approves a discount above 15%?**\nA: The Sales Manager, up to 30%."
    assert split_sentences(text) == ["Q: Who approves a discount above 15%? A: The Sales Manager, up to 30%."]


def test_split_sentences_drops_fragments():
    """Fix 3: the tail of a sentence cut at a chunk boundary is not a quote."""
    assert split_sentences("handoff record.\nThis is a full sentence with enough words in it.") == [
        "This is a full sentence with enough words in it."]


@pytest.mark.parametrize("question, days", [
    ("What should I do in my first week?", 7),
    ("first two weeks in Berlin", 14),
    ("my first month in Munich", 30),
    ("the first 90 days", 90),
    ("first 45 days", 45),
    ("what is the leave policy", None),
])
def test_time_window(question, days):
    assert time_window(normalise(question)) == days


# ---------------- M3: rules and quotes ----------------

def test_approval_question_answered_from_phase3_rules(kora):
    a = kora.answer("Who approves purchases above 10000 euros?")
    assert a.status == "answered"                       # "euros" matches "€" (fix 5)
    assert a.lines[0].kind == "rule"
    assert "APR-39BC5583" in a.rule_ids
    assert "CFO" in a.lines[0].detail


def test_generic_rule_does_not_win_every_approval_question(kora):
    """'Direct manager approves.' used to match every 'who approves' question."""
    a = kora.answer("Who approves a customer implementation plan?")
    assert a.rule_ids[0] == "APR-79C52990"
    assert "Implementation Manager" in a.lines[0].detail


def test_no_quote_repeats_a_rule(kora):
    a = kora.answer("Who approves a customer implementation plan?")
    texts = [normalise(l.text) for l in a.lines]
    assert not any(t1 != t2 and t1 in t2 for t1 in texts for t2 in texts)


def test_citations_carry_full_source_identity(kora):
    a = kora.answer("Who approves purchases above 10000 euros?")
    for c in a.citations:
        assert c.document_id and c.version and c.status and c.chunk_id
        assert c.source_filename.endswith(".md")
        assert c.effective_date != "unknown"


# ---------------- M4: conflicts, gaps, history ----------------

def test_conflict_shows_both_sides_and_picks_no_winner(kora):
    a = kora.answer("Are there conflicting instructions about security training?")
    assert a.status == "conflict"
    sides = [l for l in a.lines if l.kind == "conflict_side"]
    assert len(sides) == 2
    statuses = {a.citations[l.citation - 1].status for l in sides}
    assert statuses == {"Active", "Archived"}
    assert any("does not pick one" in n for n in a.notes)


def test_known_gap_is_surfaced(kora):
    a = kora.answer("What should the Implementation Consultant do if the AE's handoff record "
                    "is missing the go-live date?")
    assert any("GAP-BDD5A0AD" in n for n in a.notes)
    gap = [l.text for l in a.lines if l.kind == "gap"]
    assert any("does not specify" in t for t in gap)          # the whole gap paragraph is shown
    assert any("no formal fallback procedure" in t for t in gap)


def test_history_note_only_for_quoted_documents(kora):
    """Fix 6: no 'older Customer Onboarding version' note on an incident answer."""
    a = kora.answer("What is the process for escalating a production incident?")
    assert "PROC-IMPL-001" not in _docs(a)
    assert not any("PROC-IMPL-001-V1" in n for n in a.notes)


def test_version_question_lists_step_changes_from_both_versions(kora):
    a = kora.answer("How did the customer onboarding process change from the previous version?")
    changes = [l for l in a.lines if l.kind == "change"]
    assert len(changes) == 6
    versions = {(c.document_id, c.version, c.status) for c in a.citations}
    assert ("PROC-IMPL-001", "2.0", "Active") in versions
    assert ("PROC-IMPL-001", "1.0", "Archived") in versions


# ---------------- M4: abstention ----------------

def test_off_topic_question_is_not_found(kora):
    a = kora.answer("What is the company policy on bringing pets to the office?")
    assert a.status == "not_found"
    assert a.lines == []
    assert a.citations, "closest sources are still shown"


def test_partly_documented_question_flags_what_is_missing(kora):
    """'contractors' is in the knowledge base but not in the leave policy:
    flagged. 'Austin' IS covered: the policy says 'US-based employees (LOC-AUS)'."""
    a = kora.answer("What is the parental leave policy for contractors in Austin?")
    assert a.status == "partial"
    assert "contractors" in a.uncovered_terms and "austin" not in a.uncovered_terms
    assert "PROC-IT-002" not in _docs(a)       # fix 1: no Data Security quote about contractors


# ---------------- M4: Germany and communication ----------------

def test_germany_first_week_lists_the_checklist(kora):
    a = kora.answer("What should a new employee moving to Germany prepare during their first week?")
    items = [l for l in a.lines if l.kind == "checklist"]
    assert len(items) == 5
    assert _docs(a) == {"DOC-CARE-003"}
    assert any("not legal, immigration, tax" in n for n in a.notes)
    assert any("Address Registration" in n and "official guidance" in n for n in a.notes)
    assert a.status == "answered"


def test_germany_first_month_is_not_discarded_by_word_check(kora):
    """Bug found in M4: 'finish' and 'Munich' aren't in the checklist, and the
    word-coverage rule threw the whole checklist answer away."""
    a = kora.answer("What do I need to finish in my first month in Munich?")
    texts = [l.text for l in a.lines if l.kind == "checklist"]
    assert "Bank account opened" in texts
    assert a.status == "answered"


def test_tone_question_quotes_the_country_profile(kora):
    a = kora.answer("What tone should I use when writing to colleagues in Germany?")
    assert _docs(a) == {"DOC-COMM-002"}
    assert a.lines[0].text.startswith("Directness:")
    assert any("individual" in n and "preference" in n for n in a.notes)
    assert not any("not legal, immigration" in n for n in a.notes)


def test_tone_profile_line_matched_on_its_label(kora):
    a = kora.answer("How should I phrase feedback to my manager in Bengaluru?")
    assert a.lines[0].text.startswith("Feedback:")
    assert _docs(a) == {"DOC-COMM-003"}


# ---------------- faithfulness: nothing invented ----------------

FAITHFULNESS_QUESTIONS = [
    "Who approves purchases above 10000 euros?",
    "Who approves a customer implementation plan?",
    "Who approves a discount of 20%?",
    "Are there conflicting instructions about security training?",
    "What should the Implementation Consultant do if the AE's handoff record is missing the go-live date?",
    "What is the parental leave policy for contractors in Austin?",
    "What should a new employee moving to Germany prepare during their first week?",
    "What do I need to finish in my first month in Munich?",
    "How did the customer onboarding process change from the previous version?",
    "What tone should I use when writing to colleagues in Germany?",
    "What is the process for escalating a production incident?",
    "How do I get access to the production analytics environment?",
    "What does NexView do?",
]


@pytest.mark.parametrize("question", FAITHFULNESS_QUESTIONS)
def test_every_answer_line_appears_in_the_chunk_it_cites(kora, question):
    """KORA writes no sentences of its own: every line's text must be found in
    its cited chunk (compared on letters and digits, like Phase 3's loader)."""
    a = kora.answer(question)
    by_number = {c.number: c for c in a.citations}
    for line in a.lines:
        if line.citation is None:
            continue                                   # e.g. employees.csv side of a conflict
        chunk = kora._chunks[by_number[line.citation].chunk_id]
        text = line.text.split(" now: ", 1)[-1] if line.kind == "change" else line.text
        key = normalise(text)
        assert key[:40] in normalise(chunk.chunk_text), (question, line.text, chunk.chunk_id)


def test_answer_log_is_json_ready(kora):
    record = kora.answer("Who approves purchases above 10000 euros?").to_log()
    json.dumps(record)
    assert record["status"] and record["retrieval"]["elapsed_ms"] >= 0


# ---------------- runtime safety (precision first) ----------------

def test_runtime_verifier_removes_a_fabricated_line(kora, monkeypatch):
    """Even if a future bug produced a sentence that is NOT in its source,
    it must never reach the user: the answer drops it and records why."""
    real = kora._best_sentences

    def with_fake(question, outcome, used):
        out = real(question, outcome, used)
        chunk_id = out[0][0]
        return [(chunk_id, "The CFO approves everything above five euros.", 0.9, 0.9)] + out
    monkeypatch.setattr(kora, "_best_sentences", with_fake)
    a = kora.answer("Who approves purchases above 10000 euros?")
    assert all("five euros" not in l.text for l in a.lines)
    assert a.dropped_lines and a.dropped_lines[0]["text"].startswith("The CFO approves everything")
    assert any("could not be verified" in n for n in a.notes)


def test_citation_numbers_point_at_the_line_source(kora):
    """Numbers are assigned once, after verification, in line order."""
    for q in ["How did the customer onboarding process change from the previous version?",
              "Are there conflicting instructions about security training?"]:
        a = kora.answer(q)
        numbers = {c.number: c.chunk_id for c in a.citations}
        assert sorted(numbers) == list(range(1, len(numbers) + 1))
        for line in a.lines:
            assert line.citation is None or numbers[line.citation] == line.chunk_id
            assert "{ref}" not in line.detail


def test_later_quotes_must_be_comparable_to_the_best(kora):
    """'Who deploys code to production?' used to add 'Move a customer from
    signed contract to live production usage.' -- filler matching one word."""
    a = kora.answer("Who deploys code to production?")
    assert any("DevOps Engineer deploys to production" in l.text for l in a.lines)
    assert not any("signed contract to live production" in l.text for l in a.lines)


def test_document_listing_lists_describing_documents_and_notes_related_ones(kora):
    a = kora.answer("Which documents describe the customer onboarding process?")
    docs = [l for l in a.lines if l.kind == "document"]
    assert [l.chunk_id.split("_")[0] for l in docs] == ["PROC-IMPL-001"]
    assert any("header names the process" in n and "DOC-DEPT-IMPL" in n for n in a.notes)


def test_metadata_conflict_not_shown_unless_asked(kora):
    a = kora.answer("Does the archived onboarding document still apply to new customers?")
    assert a.status != "conflict"


def test_short_fact_question_shows_the_known_count_conflict(kora):
    a = kora.answer("How many office locations does Nexora have?")
    assert a.status == "conflict"
    assert "CONF-91EBA1C7" in a.retrieval.plan.conflicts


def test_unknown_subject_abstains_even_with_a_strong_context_match(kora):
    """'pets' appears nowhere in the knowledge base; 'office' matching well
    is not an answer."""
    a = kora.answer("What is the company policy on bringing pets to the office?")
    assert a.status == "not_found" and "pets" in a.notes[0]


@pytest.mark.parametrize("question", [
    "Who founded the Munich office canteen?",                       # unknown subject, context words match
    "What is the company's policy on stock options for employees?",  # homonym: inventory 'stock'
])
def test_weak_context_matches_abstain(kora, question):
    """Found on unseen questions (held-out v2): a weak match on context words
    must not become a 'partial' answer. Precision first."""
    assert kora.answer(question).status == "not_found"


# ---------------- simplification: word list + sentence rescue (M5b final) ----------------

def test_company_word_list_answers_everyday_wording(kora):
    a = kora.answer("How quickly do I get my money back after paying for a business trip myself?")
    assert a.status == "answered" and "next payroll cycle" in a.lines[0].text
    assert any(n.startswith("Searched with the company word list") and "trip -> travel" in n for n in a.notes)
    assert a.decision["synonyms"] == {"trip": ["travel"], "money back": ["reimbursed", "reimbursement"]}


def test_word_list_never_changes_the_question_or_quotes(kora):
    a = kora.answer("On which day are new features usually pushed live?")
    assert a.question == "On which day are new features usually pushed live?"
    assert "Wednesday" in a.lines[0].text
    assert all(kora.verify_line(l) for l in a.lines)          # still word-for-word from the source


def test_word_list_does_not_rescue_unknown_subjects(kora):
    """The list only adds words; 'pets' still isn't in the knowledge base."""
    assert kora.answer("Can I bring my pets on a business trip?").status == "not_found"


def test_low_ranked_chunk_with_the_exact_sentence_can_be_quoted(kora):
    """Q37: the answer sentence sits under '## Channels' in a chunk that
    ranked low overall; it covers 'channel' + 'quick', more than any other."""
    a = kora.answer("Which channel should I use for a quick question to a colleague?")
    assert a.status == "answered" and a.lines[0].text.startswith("Chat (internal messaging tool)")


def test_word_list_config_is_validated():
    from src.retrieval.config import ConfigError, parse_config
    cfg = parse_config({"synonyms": {"_comment": "ignored", "Money-Back": ["Reimbursed"]}})
    assert cfg.synonyms == {"money back": ["reimbursed"]}
    with pytest.raises(ConfigError, match="synonyms"):
        parse_config({"synonyms": {"trip": "travel"}})          # must be a list
    with pytest.raises(ConfigError, match="synonyms"):
        parse_config({"synonyms": {"trip": []}})


def test_no_word_list_means_no_expansion(intel):
    from src.retrieval.config import parse_config
    k = KoraAnswerer.from_intelligence(intel, parse_config({}))
    assert k.expand("my business trip") == ("my business trip", {})


# ---------------- held-out v4 lessons (v4 is now SPENT; these are regression tests) ----------------

@pytest.mark.parametrize("question, word", [
    ("How many leave days do employees in the Madrid office get?", "madrid"),   # unknown place
    ("How do I book a meeting room in Bengaluru?", "room"),                       # unknown subject
    ("How many days of annual leave do part-time employees in London get?", "part time"),
])
def test_a_word_the_documents_never_use_is_never_guessed(kora, question, word):
    a = kora.answer(question)
    assert a.status == "not_found" and a.lines == []
    assert word in a.decision["unknown"] and "company word list" in a.decision["rule_text"]


def test_a_place_is_matched_by_its_code(kora):
    """The leave policy never says 'London'; it says 'UK-based employees (LOC-LON)'."""
    a = kora.answer("How many days of annual leave do employees in London get?")
    assert a.status == "answered" and a.lines[0].text.startswith("UK-based employees (LOC-LON): 25 days")


def test_hyphenated_words_count_as_one(kora):
    from src.retrieval.answer import _COMPOUND
    assert _COMPOUND.findall("part-time staff on PROC-IT-001 after go-live") == ["part-time", "go-live"]


# ---------------- UT's own questions (held-out v5, now spent; regression tests) ----------------

@pytest.mark.parametrize("question", [
    "What is my salary structure?",                 # "salary" + "structure" scattered over two lines
    "What is the procedure to change the team?",    # every word 'generic' -> nothing was checked
    "If I come late to the office will my salary be cut?",
    "If I do overtime do I get paid extra?",
])
def test_undocumented_hr_questions_are_refused(kora, question):
    a = kora.answer(question)
    assert a.status == "not_found" and a.lines == []


def test_all_generic_question_still_checks_its_words(kora):
    a = kora.answer("What is the procedure to change the team?")
    assert {"procedure", "change", "team"} <= set(a.decision.get("important", []))


def test_who_do_i_contact_is_answered_from_the_policy(kora):
    a = kora.answer("Who do I contact for reimbursement?")
    assert a.status == "answered" and "reimbursed" in a.lines[0].text


def test_gap_lines_must_pass_the_same_checks_as_quotes(kora):
    """A declared-gap paragraph used to count as a 'structured fact' and skip
    every check, so an unrelated question could be answered with it."""
    from src.retrieval.answer import STRUCTURED_KINDS
    assert "gap" not in STRUCTURED_KINDS
    a = kora.answer("What should the Implementation Consultant do if the AE's handoff record "
                    "is missing the go-live date?")
    assert any(l.kind == "gap" for l in a.lines)
