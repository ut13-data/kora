"""
Phase 5 M5: the screens and the endpoints they added.

  * pages need a sign-in; Settings only for admins
  * "My week" shows your own context, with nothing from documents you can't read
  * settings are validated, backed up, applied at once, and restorable
The app works on a temporary copy of the config, index and log.
"""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from src.index import Kora, Paths
    from src.web.app import create_app
    if not Path("data/raw").exists():
        pytest.skip("Nexora data not found in data/raw")
    tmp = tmp_path_factory.mktemp("ui")
    config = tmp / "config" / "kora_config.json"
    config.parent.mkdir()
    shutil.copy("config/kora_config.json", config)
    paths = Paths(raw_dir=Path("data/raw"), processed_dir=tmp / "processed", chunks_dir=tmp / "chunks",
                  intelligence_dir=tmp / "intelligence", context_dir=Path("data/context"),
                  manifest=tmp / "manifest.json", config=config, log_dir=tmp / "logs")
    kora = Kora.open(paths, channel="api")
    return {"app": create_app(kora=kora, secret="s", dev_passcode="pass"), "kora": kora, "paths": paths}


def client(setup, emp=None):
    c = TestClient(setup["app"], follow_redirects=False)
    if emp:
        assert c.post("/api/login", json={"employee_id": emp, "passcode": "pass"}).status_code == 200
    return c


# ---------------- pages ----------------

def test_login_page_and_static_files(setup):
    c = client(setup)
    page = c.get("/login")
    assert page.status_code == 200 and "Development sign-in" in page.text
    for f in ("kora.css", "kora.js", "ask.js", "explore.js", "me.js", "admin.js", "login.js", "document.js"):
        assert c.get(f"/static/{f}").status_code == 200, f


@pytest.mark.parametrize("path", ["/ask", "/explore", "/me", "/admin", "/documents/PROC-HR-002"])
def test_pages_need_sign_in(setup, path):
    r = client(setup).get(path)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_pages_greet_the_signed_in_employee(setup):
    c = client(setup, "EMP-0040")
    page = c.get("/ask")
    assert page.status_code == 200 and "Hi Manoj" in page.text and "Settings" not in page.text
    for path in ("/explore", "/me", "/documents/PROC-HR-002"):
        assert c.get(path).status_code == 200


def test_settings_page_is_for_admins(setup):
    assert client(setup, "EMP-0040").get("/admin").headers["location"] == "/ask"
    page = client(setup, "EMP-0120").get("/admin")
    assert page.status_code == 200 and ">Settings<" in page.text


def test_ui_questions_are_logged_as_ui(setup):
    client(setup, "EMP-0040").post("/api/ask", json={"question": "Who owns the procurement process?", "channel": "ui"})
    last = json.loads((setup["paths"].log_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["channel"] == "ui"
    bad = client(setup, "EMP-0040").post("/api/ask", json={"question": "x", "channel": "cli"})
    assert bad.status_code == 422


# ---------------- my week ----------------

def test_my_week_for_a_relocating_new_joiner(setup):
    me = client(setup, "EMP-0085").get("/api/me/context").json()
    assert me["employee_id"] == "EMP-0085" and me["joining_stage"] == "week_2" and me["as_of"]
    assert me["germany_care"]["eligibility"]["status"] == "active"
    saved = json.loads((setup["paths"].intelligence_dir / "employee_context.json").read_text(encoding="utf-8"))
    raw = next(e for e in saved["employees"] if e["employee_id"] == "EMP-0085")
    assert len(me["germany_care"]["plan"]["items"]) == len(raw["germany_care"]["plan"]["items"]) == 13  # nothing dropped
    assert "not legal" in me["germany_care"]["plan"]["caveat"]
    assert all(i["status"] == "not_reported" for i in me["germany_care"]["plan"]["items"])   # never ticked by KORA
    assert "not about this person" in me["communication"]["basis"]


def test_my_week_never_names_hidden_documents(setup):
    """Customer Success employees appear in the invoice process's steps, but may not read it."""
    for emp in ("EMP-0090", "EMP-0040"):
        text = json.dumps(client(setup, emp).get("/api/me/context").json())
        for hidden in ("PROC-FIN-003", "Invoice and Payment", "DOC-DEPT-FIN"):
            assert hidden not in text, (emp, hidden)
    finance = json.dumps(client(setup, "EMP-0113").get("/api/me/context").json())
    assert "PROC-FIN" in finance


# ---------------- settings ----------------

def _config(setup):
    return json.loads(setup["paths"].config.read_text(encoding="utf-8"))


def test_settings_endpoints_are_admin_only(setup):
    c = client(setup, "EMP-0040")
    for method, url in (("get", "/api/admin/config/schema"), ("get", "/api/admin/config/backups")):
        assert getattr(c, method)(url).status_code == 403
    assert c.put("/api/admin/config", json={}).status_code == 403
    assert c.post("/api/admin/config/validate", json={}).status_code == 403


def test_schema_offers_thresholds_and_columns(setup):
    s = client(setup, "EMP-0120").get("/api/admin/config/schema").json()
    assert s["thresholds"]["top_k"]["min"] == 1 and "manager_id" in s["employee_columns"]
    assert "first_name" not in s["employee_columns"]


@pytest.mark.parametrize("change,problem", [
    (lambda c: c["thresholds"].update(top_k=999), "top_k"),
    (lambda c: c["admins"].update(employees=["EMP-0126"]), "lose admin access"),
    (lambda c: c["admins"].update(employees=["EMP-0120", "EMP-9999"]), "unknown employee"),
    (lambda c: c["access"]["rules"].update({"PROC-X": ["emp:EMP-9999"]}), "unknown employee"),
    (lambda c: c.update(nonsense=True), "unknown key"),
])
def test_bad_settings_are_refused_and_never_written(setup, change, problem):
    before = setup["paths"].config.read_text(encoding="utf-8")
    raw = _config(setup)
    change(raw)
    c = client(setup, "EMP-0120")
    v = c.post("/api/admin/config/validate", json=raw).json()
    assert not v["valid"] and any(problem in p for p in v["problems"]), v
    r = c.put("/api/admin/config", json=raw)
    assert r.status_code == 422 and setup["paths"].config.read_text(encoding="utf-8") == before


def test_save_backs_up_applies_at_once_and_restores(setup):
    c = client(setup, "EMP-0120")
    original = _config(setup)
    raw = _config(setup)
    raw["synonyms"]["holiday"] = ["leave"]
    r = c.put("/api/admin/config", json=raw).json()
    assert r["saved"] and r["backup"]
    assert _config(setup)["synonyms"]["holiday"] == ["leave"]
    assert setup["kora"].answerer.config.synonyms["holiday"] == ["leave"]          # KORA reloaded
    backups = c.get("/api/admin/config/backups").json()
    assert backups[0]["name"] == r["backup"]
    restored = c.post("/api/admin/config/restore", json={"name": r["backup"]}).json()
    assert restored["saved"] and _config(setup) == original
    assert "holiday" not in setup["kora"].answerer.config.synonyms


@pytest.mark.parametrize("name", ["../kora_config.json", "kora_config.json", "kora_config.2026.json", "x"])
def test_restore_only_accepts_backup_names(setup, name):
    assert client(setup, "EMP-0120").post("/api/admin/config/restore", json={"name": name}).status_code == 404
