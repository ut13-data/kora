"""Phase 3, Milestone 4: Germany new-joiner care, communication tone and employee context,
on the real Nexora data with the default as_of date (2025-04-22). The `intel` fixture lives in conftest.py."""

import json
from datetime import date

import pytest

from src.intelligence.communication import build_communication
from src.intelligence.employee_context import add_working_days
from src.intelligence.germany import build_germany_care


def _employee(intel, emp_id):
    return intel.employee(emp_id)


# ---------------- Germany care ----------------

def test_items_keep_nexora_guidance_and_official_requirements_apart(intel):
    items = intel.germany.items
    address = items["GER-ADDRESS-REGISTRATION"]
    assert address.official_guidance_note.lower().startswith("check the current official guidance")
    health = items["GER-HEALTH-INSURANCE"]
    assert any("mandatory" in s for s in health.external_requirements)
    assert not any("mandatory" in s for s in health.nexora_guidance)


def test_dependencies_come_from_the_text(intel):
    deps = {i: [d["item_id"] for d in item.depends_on] for i, item in intel.germany.items.items()}
    assert deps["GER-ADDRESS-REGISTRATION"] == ["GER-HOUSING"]
    assert deps["GER-TAX-REGISTRATION"] == ["GER-ADDRESS-REGISTRATION"]
    assert "GER-HOUSING" in deps["GER-BANK-ACCOUNT"]
    order = intel.germany.order
    assert order.index("GER-HOUSING") < order.index("GER-ADDRESS-REGISTRATION") < order.index("GER-TAX-REGISTRATION")


def test_commonly_forgotten_items(intel):
    ranks = {i: item.commonly_forgotten["rank"] for i, item in intel.germany.items.items() if item.commonly_forgotten}
    assert ranks == {"GER-ADDRESS-REGISTRATION": 1, "GER-INTERNET-SETUP": 2, "GER-FINDING-A-GENERAL-PRACTITIONER": 3}


def test_a_sentence_saying_what_hr_cannot_do_is_not_a_contact(intel):
    assert intel.germany.items["GER-TAX-REGISTRATION"].nexora_contacts == []


def test_munich_only_contacts_are_a_gap_for_berlin(intel):
    berlin = [g for g in intel.germany.gaps if g["gap_type"] == "checklist_contact_not_local"]
    assert {g["item_id"] for g in berlin} == {"GER-HOUSING", "GER-EMERGENCY-CONTACTS"}
    assert all(g["location_id"] == "LOC-BER" for g in berlin)


def test_eligibility_comes_from_employees_csv(intel):
    status = {e: intel.germany.eligibility(intel.entities.get("employee", e), intel.entities)
              for e in ("EMP-0085", "EMP-0023", "EMP-0031", "EMP-0001")}
    assert (status["EMP-0085"]["status"], status["EMP-0085"]["stage"]) == ("active", "week_2")
    assert (status["EMP-0023"]["status"], status["EMP-0023"]["stage"]) == ("active", "month_2")
    assert status["EMP-0031"]["status"] == "past_program_window"
    assert status["EMP-0001"]["status"] == "not_applicable"            # Austin


def test_nothing_is_complete_until_reported(intel):
    plan = _employee(intel, "EMP-0085")["germany_care"]["plan"]
    assert plan["progress_reported"] is False
    assert {i["status"] for i in plan["items"]} == {"not_reported"}
    assert plan["next_steps"][0] == "GER-HOUSING"                     # address registration waits for housing


def test_reported_progress_unblocks_next_steps(intel, tmp_path):
    progress = tmp_path / "checklist_progress.json"
    progress.write_text(json.dumps({"employees": {"EMP-0085": {"completed": {"GER-HOUSING": "2025-04-12"}}}}))
    care = build_germany_care(intel.phase2, intel.entities, date(2025, 4, 22), progress)
    plan = care.plan("EMP-0085", 14, "Munich")
    status = {i["item_id"]: i["status"] for i in plan["items"]}
    assert status["GER-HOUSING"] == "reported_complete"
    assert "GER-ADDRESS-REGISTRATION" in plan["next_steps"]
    assert plan["possibly_forgotten"][0] == "GER-ADDRESS-REGISTRATION"   # commonly forgotten #1, past its pace


# ---------------- communication ----------------

def test_profiles_are_linked_to_offices(intel):
    profiles = {p.profile_id: p for p in intel.communication.profiles}
    assert profiles["COMM-PROFILE-GERMANY"].location_ids == ["LOC-BER", "LOC-MUC"]
    assert profiles["COMM-PROFILE-INTERNATIONAL-NEUTRAL"].is_default
    assert len(profiles) == 5


def test_every_place_profile_uses_hedged_language(intel):
    check = intel.communication.to_dict(intel.entities)["hedging_check"]
    assert len(check["profiles_checked"]) == 4 and check["unhedged_statements"] == []


def test_four_layer_framework(intel):
    fw = intel.communication.framework
    assert [layer["layer"] for layer in fw["layers"]] == [
        "Fact", "Company Preference", "Cultural Communication Guidance", "Individual Preference"]
    assert "overrides" in fw["precedence"]


@pytest.mark.parametrize("culture", ["COMM-PROFILE-GERMANY", "COMM-PROFILE-INDIA", "COMM-PROFILE-UNITED-STATES",
                                     "COMM-PROFILE-UNITED-KINGDOM", "COMM-PROFILE-INTERNATIONAL-NEUTRAL"])
@pytest.mark.parametrize("tone", ["neutral", "direct", "friendly"])
def test_tone_never_changes_the_fact(intel, culture, tone):
    fact = "Please complete the security training by Friday."
    out = intel.communication.present(fact, culture, tone)
    assert out["fact"] == fact and out["rendered"].startswith(fact) and out["fact_unchanged"]


def test_direct_style_asks_for_the_date_instead_of_inventing_it(intel):
    out = intel.communication.present("Please complete the security training by Friday.", "COMM-PROFILE-GERMANY", "direct")
    assert out["rendered"] == "Please complete the security training by Friday."
    assert any(c["check"] == "direct_style_date" and c["result"] is False for c in out["checks"])


def test_friendly_tone_adds_only_a_documented_phrase(intel):
    out = intel.communication.present("Please complete the security training by Friday.", None, "friendly")
    assert out["added_phrase"]["source_document"].startswith("DOC-COMM-")
    with pytest.raises(ValueError):
        intel.communication.present("x", None, "sarcastic")


def test_individual_preference_overrides_the_location_profile(intel, tmp_path):
    prefs = tmp_path / "communication_preferences.json"
    prefs.write_text(json.dumps({"employees": {"EMP-0085": {"preferred_tone": "relationship_oriented", "set_by": "self"}}}))
    comms = build_communication(intel.phase2, intel.entities, prefs)
    assert comms.for_employee("EMP-0085", "LOC-MUC")["effective_tone"] == "relationship_oriented"
    default = intel.communication.for_employee("EMP-0085", "LOC-MUC")
    assert default["individual_preference"] == "unknown" and "hypothesis" in default["basis"]


# ---------------- employee context ----------------

def test_working_days_skip_weekends():
    assert add_working_days(date(2025, 4, 9), 10) == date(2025, 4, 22)   # Wed + 10 working days


def test_new_joiner_context_is_personal_and_traceable(intel):
    e = _employee(intel, "EMP-0085")
    assert (e["role"]["role_id"], e["location"]["city"], e["joining_stage"]) == ("ROLE-024", "Munich", "week_2")
    assert e["manager"]["employee_id"] == "EMP-0077"
    training = next(s for s in e["onboarding"]["steps"] if s["title"] == "Security training")
    assert training["date"] == "2025-04-22" and "working days" in training["date_basis"]
    assert "CONF-" in e["onboarding"]["note"]
    assert all(x["your_manager"]["employee_id"] == "EMP-0077" for x in e["approvals"]["your_manager_approves"])
    reading = [r["document_id"] for r in e["reading_list"]]
    assert "DOC-CARE-001" in reading and "DOC-COMM-002" in reading
    assert all(r["why"] for r in e["reading_list"])


def test_hypothetical_profile_is_labelled_and_finds_the_real_approver(intel):
    [h] = intel.employee_context["hypothetical_examples"]
    assert h["hypothetical"] and h["employee_id"] is None and h["joining_stage"] == "week_2"
    approvers = {p["employee_id"] for x in h["approvals"]["you_may_need_approval_from"]
                 for a in x["approvers"] for p in a["people"]}
    assert "EMP-0066" in approvers                                     # the actual Data & Analytics Manager


def test_context_never_contradicts_employees_csv(intel):
    for e in intel.employee_context["employees"]:
        row = intel.entities.get("employee", e["employee_id"]).attributes
        assert (e["role"]["role_id"], e["department"]["dept_id"], e["location"]["location_id"]) == \
               (row["role_id"], row["dept_id"], row["location_id"])


def test_eligible_for_links_only_relocating_employees(intel):
    links = [r for r in intel.relationships if r.relationship == "eligible_for"]
    assert links
    for r in links:
        assert intel.entities.get("employee", r.source_id).attributes["international_relocation_germany"] == "Yes"
