"""Phase 4 M2: router (where to look) and KoraRetriever (router + search)."""

import json

import pytest

from src.chunking.chunker import Chunk
from src.retrieval.keyword import build_keyword_index
from src.retrieval.retrieve import KoraRetriever
from src.retrieval.config import load_config
from src.retrieval.router import Router, _content_words, normalise
from src.retrieval.search import ChunkRetriever
from src.retrieval.tfidf import build_tfidf_index


@pytest.fixture(scope="module")
def router(intel):
    return Router.from_outputs(intel.outputs, load_config())


@pytest.fixture(scope="module")
def kora(intel):
    return KoraRetriever.from_intelligence(intel, load_config())


def _ids(plan, entity_type):
    return [e.entity_id for e in plan.entities if e.entity_type == entity_type]


# ---------------- normalise ----------------

def test_normalise_handles_ampersand_slash_and_case():
    assert normalise("Data & Analytics") == "data and analytics"
    assert normalise("IT / Security") == "it security"
    assert normalise("First-Week Essentials — Germany") == "first week essentials germany"


# ---------------- intents ----------------

def test_relocation_intent_routes_to_new_joiner_documents(router):
    plan = router.route("What should a new employee moving to Germany prepare during their first week?")
    assert "relocation" in plan.intents
    assert plan.conflicts == []           # bug: filler words "new/their/first" matched a conflict
    assert set(plan.candidate_documents) >= {"DOC-CARE-002", "DOC-CARE-003", "DOC-CARE-004"}


def test_german_city_switches_on_relocation_intent(router):
    plan = router.route("I just joined in Munich, what do I need to do?")
    assert "LOC-MUC" in _ids(plan, "location")
    assert "relocation" in plan.intents
    assert any("country germany named" in r for r in plan.reasons)


def test_communication_intent_routes_to_tone_documents(router):
    plan = router.route("What tone should I use with colleagues in the UK?")
    assert "communication" in plan.intents
    assert "DOC-COMM-005" in plan.candidate_documents


# ---------------- topic matching ----------------

def test_strong_topic_match_narrows_to_its_documents(router):
    plan = router.route("Which documents describe the customer onboarding process?")
    assert "TOPIC-CUSTOMER-ONBOARDING" in _ids(plan, "topic")
    assert plan.candidate_documents == ["PROC-IMPL-001"]


def test_weak_match_ignored_when_a_strong_match_exists(router):
    """'sales' is a single generic department word (weak); 'expense policy'
    is specific (strong). Only the specific topic should narrow the search."""
    plan = router.route("What is the expense policy for sales trips?")
    strengths = {e.entity_id: e.strength for e in plan.entities if e.entity_type == "topic"}
    assert strengths.get("TOPIC-EXPENSE-POLICY") == "strong"
    assert strengths.get("TOPIC-SALES") == "weak"
    assert plan.candidate_documents == ["PROC-FIN-001"]


def test_weak_match_used_when_nothing_strong_matched(router):
    plan = router.route("Who approves a customer implementation plan?")
    assert "approval" in plan.intents
    assert "PROC-IMPL-001" in plan.candidate_documents
    assert any("broad department topic" in r for r in plan.reasons)


def test_product_alias_matches(router):
    plan = router.route("How does NexView refresh dashboards?")
    assert "TOPIC-NEXVIEW-ANALYTICS" in _ids(plan, "topic")
    assert "DOC-PROD-NEXVIEW" in plan.candidate_documents


def test_role_acronym_is_case_sensitive(router):
    assert "ROLE-003" in _ids(router.route("What does an AE hand over after a deal?"), "role")
    # lowercase "ae" inside a normal word or sentence must not match
    assert "ROLE-003" not in _ids(router.route("the aesthetics of the dashboard"), "role")


def test_roles_are_recorded_but_do_not_narrow(router):
    plan = router.route("What does a Solutions Architect do?")
    assert "ROLE-023" in _ids(plan, "role")
    assert plan.candidate_documents == []


def test_nothing_recognised_searches_everything(router):
    plan = router.route("zzqqxx vvbbnn")
    assert plan.candidate_documents == []
    assert any("whole knowledge base" in r for r in plan.reasons)


def test_candidate_documents_have_no_duplicates(router):
    plan = router.route("what does PROC-IMPL-001 say about kickoff")
    assert len(plan.candidate_documents) == len(set(plan.candidate_documents))


# ---------------- archived documents (decision C) ----------------

def test_archived_excluded_by_default(router):
    plan = router.route("What are the customer onboarding steps?")
    assert plan.include_archived is False


def test_archived_included_for_version_questions(router):
    plan = router.route("How did the customer onboarding process change from the previous version?")
    assert "version" in plan.intents
    assert plan.include_archived is True


def test_archived_included_for_conflict_questions(router):
    plan = router.route("Are there conflicting instructions about security training?")
    assert "conflict" in plan.intents
    assert plan.include_archived is True


def test_naming_an_archived_document_includes_archived(router):
    plan = router.route("What did PROC-HR-001-ADDENDUM require?")
    assert plan.include_archived is True
    assert "PROC-HR-001-ADDENDUM" in plan.candidate_documents


def test_archived_file_with_borrowed_header_id_is_reachable(router):
    """PROC-IMPL-001-V1's chunks carry the header ID PROC-IMPL-001, so naming
    the V1 ID must also put PROC-IMPL-001 in the candidates."""
    plan = router.route("Show me PROC-IMPL-001-V1")
    assert plan.include_archived is True
    assert "PROC-IMPL-001" in plan.candidate_documents


# ---------------- KoraRetriever: router + search ----------------

def test_default_retrieval_has_no_archived_chunks_but_notes_history(kora):
    outcome = kora.retrieve("What are the customer onboarding steps?")
    assert outcome.results
    assert all(r.chunk.status != "Archived" for r in outcome.results)
    assert any(n.archived_document_id == "PROC-IMPL-001-V1" for n in outcome.history_notes)


def test_version_question_retrieves_archived_chunks_without_notes(kora):
    outcome = kora.retrieve("How did the customer onboarding process change from the previous version?")
    assert any(r.chunk.status == "Archived" for r in outcome.results)
    assert outcome.history_notes == []


def test_routed_tier_leaves_two_slots_for_whole_corpus_search(kora):
    """Bug 1 from M2: a weak match ('security') used to fill all 5 slots
    with IT documents and push out the archived security-training addendum."""
    outcome = kora.retrieve("Are there conflicting instructions about security training?", top_k=5)
    routed = [r for r in outcome.results if r.matched_by == "routed"]
    assert len(routed) <= 3
    assert any(r.chunk.document_id == "PROC-HR-001-ADDENDUM" for r in outcome.results)


def test_germany_question_puts_checklists_first(kora):
    outcome = kora.retrieve("What should a new employee moving to Germany prepare during their first week?")
    assert outcome.results[0].matched_by == "routed"
    assert outcome.results[0].chunk.document_id.startswith("DOC-CARE-")


def test_log_record_is_json_ready(kora):
    record = kora.retrieve("Who approves a customer implementation plan?").to_log()
    json.dumps(record)                     # raises if anything isn't serialisable
    assert record["elapsed_ms"] >= 0
    assert {"chunk_id", "document_id", "status", "tier", "score"} <= set(record["results"][0])


# ---------------- reserved slots go back to the router if unused ----------------

def _chunk(chunk_id: str, document_id: str, text: str, index: int) -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id=document_id, chunk_text=text,
                 chunk_index=index, document_title=document_id, department="Test",
                 document_type="Test", version="1.0", status="Active",
                 section="", parent_id=f"{chunk_id}_p")


def test_reserved_slots_are_not_wasted_when_only_router_documents_are_relevant():
    """Only DOC-A talks about apples. The router fills 3 slots; the 2 reserved
    slots must still be filled -- by the whole-corpus search, which finds
    DOC-A's remaining chunks on its own -- rather than left empty."""
    chunks = [_chunk(f"DOC-A_v1.0_00{i}", "DOC-A", f"apple orchard apple harvest note {i}", i)
              for i in range(5)]
    chunks.append(_chunk("DOC-B_v1.0_000", "DOC-B", "grape melon kiwi", 0))
    retriever = ChunkRetriever(keyword=build_keyword_index(chunks), tfidf=build_tfidf_index(chunks))
    results = retriever.search("apple orchard", top_k=5, document_ids=["DOC-A"])
    assert len(results) == 5
    assert [r.matched_by for r in results].count("routed") == 3
    assert all(r.chunk.document_id == "DOC-A" for r in results)


# ---------------- Fix 1: conflict-aware routing ----------------

def test_content_words_drop_stopwords_and_plural_s():
    assert _content_words("Are there conflicting instructions about security trainings?") \
        == {"security", "training"}


def test_conflict_question_pins_both_sides_first(kora):
    """The question that failed after M2: the router searched IT documents and
    missed the active rule. Phase 3 already knows this conflict."""
    outcome = kora.retrieve("Are there conflicting instructions about security training?")
    assert outcome.plan.conflicts == ["CONF-98434E88"]
    first_two = outcome.results[:2]
    assert [r.matched_by for r in first_two] == ["conflict", "conflict"]
    assert {r.chunk.document_id for r in first_two} == {"PROC-HR-001", "PROC-HR-001-ADDENDUM"}
    assert {r.chunk.status for r in first_two} == {"Active", "Archived"}


def test_conflict_tiers_still_leave_two_slots_for_whole_corpus(kora):
    outcome = kora.retrieve("Are there conflicting instructions about security training?", top_k=5)
    early = [r for r in outcome.results if r.matched_by in ("conflict", "routed")]
    assert len(early) <= 3
    assert sum(r.matched_by == "tfidf" for r in outcome.results) >= 2


def test_conflict_question_without_a_detected_conflict_says_so(router):
    plan = router.route("Are there conflicting rules about office parking?")
    assert "conflict" in plan.intents
    assert plan.conflicts == [] and plan.pinned_chunks == []
    assert any("no detected conflict matches" in r for r in plan.reasons)


def test_matched_conflict_replaces_broad_department_guess(router):
    plan = router.route("Is there a conflict about the reporting line of the Implementation Manager?")
    assert "CONF-816ACA6F" in plan.conflicts
    assert "DOC-CORP-002" in plan.candidate_documents
    assert "DOC-DEPT-IMPL" not in plan.candidate_documents      # weak 'implementation' ignored


def test_pinned_chunks_are_kept_even_when_archived_is_excluded():
    chunks = [_chunk("DOC-A_v1.0_000", "DOC-A", "old rule thirty days", 0)]
    chunks[0].status = "Archived"
    retriever = ChunkRetriever(keyword=build_keyword_index(chunks), tfidf=build_tfidf_index(chunks))
    results = retriever.search("rule", include_archived=False, pinned_chunk_ids=["DOC-A_v1.0_000"])
    assert [r.matched_by for r in results] == ["conflict"]


# ---------------- Fix 2: Phase 3 keywords as fallback ----------------

def test_keywords_find_the_right_document_without_a_topic_name(router):
    """'retention period' is not a topic name; Phase 3 keywords of PROC-IT-002 match."""
    plan = router.route("What is the retention period for customer data?")
    assert "PROC-IT-002" in plan.candidate_documents
    assert any("keywords" in r and "PROC-IT-002" in r for r in plan.reasons)


def test_keyword_match_on_archived_document_brings_in_its_successor():
    """Generic rule, tested on a minimal catalogue: an archived document found
    by keywords also brings in the active document that replaced it."""
    entities = {"entities": [
        {"key": "document:OLD-1", "entity_id": "OLD-1", "entity_type": "document", "name": "Old rule",
         "aliases": [], "attributes": {"status": "Archived"}},
        {"key": "document:NEW-1", "entity_id": "NEW-1", "entity_type": "document", "name": "New rule",
         "aliases": [], "attributes": {"status": "Active"}}]}
    topics = {"topics": [], "document_keywords": {"OLD-1": ["parking", "permit"], "NEW-1": ["other"]}}
    outdated = {"superseded_documents": [{"document_id": "OLD-1", "superseded_by": [{"document_id": "NEW-1"}]}]}
    plan = Router(entities, topics, {}, outdated).route("How do I get a parking permit?")
    assert plan.candidate_documents == ["OLD-1", "NEW-1"]
    assert any("NEW-1 replaced archived OLD-1" in r for r in plan.reasons)


def test_security_deadline_question_surfaces_the_known_conflict(kora):
    """Without the word 'conflict', a question clearly about a conflicting fact
    (3+ shared words, confident conflict) shows both sides first."""
    outcome = kora.retrieve("What is the security training deadline for new hires?")
    assert outcome.plan.conflicts == ["CONF-98434E88"]
    assert [r.matched_by for r in outcome.results[:2]] == ["conflict", "conflict"]


def test_keywords_not_used_when_a_topic_name_matched(router):
    plan = router.route("What is the expense policy for sales trips?")
    assert not any(r.startswith("candidates") and "keywords" in r for r in plan.reasons)


# ---------------- phrases, not single words ----------------

@pytest.mark.parametrize("question, is_history", [
    ("How did the customer onboarding process change from the previous version?", True),
    ("What did the archived security training addendum say?", True),
    ("What changed in v2.0 of customer onboarding?", True),
    ("What is the process when a customer reports a bug on a deprecated product version?", False),
    ("What happens if a customer wants to change the go-live date?", False),
    ("A customer reported a bug on an old configuration -- what now?", False),
])
def test_history_intent_needs_a_history_phrase(router, question, is_history):
    """Bug found in M5: single words ('version', 'change', 'old') opened
    archived documents and showed unrelated version changes."""
    plan = router.route(question)
    assert ("version" in plan.intents) is is_history


def test_conflict_matching_ignores_filler_and_document_ids(router):
    assert router.route("What should a new employee moving to Germany prepare during their first week?").conflicts == []
    assert router.route("What does PROC-IMPL-001 say about kickoff").conflicts == []


def test_conflict_matching_uses_statements_not_only_the_label(router):
    """The label is Phase 3's top TF-IDF terms and differs between library
    versions; statements are source text, identical everywhere."""
    plan = router.route("Is the 30-day deadline for new hire security training still valid?")
    assert plan.conflicts == ["CONF-98434E88"]


def test_communication_word_alone_is_not_a_tone_question(router):
    plan = router.route("Who owns customer communication about the status of a defect?")
    assert "communication" not in plan.intents and plan.tone_profiles == []
    plan = router.route("How do colleagues in the US usually communicate urgency?")
    assert plan.tone_profiles == ["DOC-COMM-004"]
