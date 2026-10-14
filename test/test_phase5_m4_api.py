"""
Phase 5 M4: the web API -- development login, asking, sources, graph, admin.

The app runs on the real Nexora files, but builds its index and writes its
log in a temporary folder (your data/ is never touched).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HIDDEN = ("PROC-FIN-003", "DOC-DEPT-FIN", "Invoice and Payment", "AP/AR")
INVOICE_Q = "Who takes part in the invoice and payment process?"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from src.index import Kora, Paths
    from src.web.app import create_app
    if not Path("data/raw").exists():
        pytest.skip("Nexora data not found in data/raw")
    tmp = tmp_path_factory.mktemp("web")
    paths = Paths(raw_dir=Path("data/raw"), processed_dir=tmp / "processed", chunks_dir=tmp / "chunks",
                  intelligence_dir=tmp / "intelligence", context_dir=Path("data/context"),
                  manifest=tmp / "manifest.json", config=Path("config/kora_config.json"), log_dir=tmp / "logs")
    kora = Kora.open(paths, channel="api")
    app = create_app(kora=kora, secret="test-secret", dev_passcode="pass")
    return {"app": app, "kora": kora, "paths": paths}


def client(setup, emp=None):
    c = TestClient(setup["app"])
    if emp:
        r = c.post("/api/login", json={"employee_id": emp, "passcode": "pass"})
        assert r.status_code == 200, r.text
    return c


def visible_text(payload, question) -> str:
    return json.dumps(payload).replace(question, "")


# ---------------- login ----------------

def test_health_is_open_and_says_dev_login(setup):
    r = client(setup).get("/api/health").json()
    assert r["index_current"] and "NOT production identity" in r["login"]


@pytest.mark.parametrize("method,url", [("post", "/api/ask"), ("get", "/api/me"), ("get", "/api/chunks/x"),
                                         ("get", "/api/documents/x"), ("get", "/api/graph/search?q=sales"),
                                         ("get", "/api/graph/entity/process:PROC-FIN-002"),
                                         ("get", "/api/admin/config")])
def test_everything_else_needs_a_login(setup, method, url):
    r = getattr(client(setup), method)(url, **({"json": {"question": "x"}} if method == "post" else {}))
    assert r.status_code == 401


def test_wrong_passcode_and_unknown_employee_get_the_same_answer(setup):
    c = client(setup)
    bad_pass = c.post("/api/login", json={"employee_id": "EMP-0040", "passcode": "nope"})
    bad_emp = c.post("/api/login", json={"employee_id": "EMP-9999", "passcode": "pass"})
    assert bad_pass.status_code == bad_emp.status_code == 401
    assert bad_pass.json() == bad_emp.json()


def test_groups_and_admin_come_from_the_server(setup):
    me = client(setup, "EMP-0040").get("/api/me").json()
    assert me["employee_id"] == "EMP-0040" and "dept:DEPT-ENG" in me["groups"] and me["admin"] is False
    assert client(setup, "EMP-0120").get("/api/me").json()["admin"] is True


def test_edited_session_cookie_is_rejected(setup):
    c = client(setup, "EMP-0040")
    cookie = c.cookies.get("kora_session")
    c.cookies.set("kora_session", cookie[:-4] + ("AAAA" if not cookie.endswith("AAAA") else "BBBB"))
    assert c.get("/api/me").status_code == 401


def test_logout(setup):
    c = client(setup, "EMP-0040")
    c.post("/api/logout")
    assert c.get("/api/me").status_code == 401


def test_picker_shows_directory_fields_only(setup):
    rows = client(setup).get("/api/login/employees").json()
    assert len(rows) == 137 and set(rows[0]) == {"employee_id", "name", "role", "department", "location"}


# ---------------- asking ----------------

def test_ask_returns_the_full_answer_and_logs_channel_api(setup):
    r = client(setup, "EMP-0040").post("/api/ask", json={"question": "Who owns the procurement process?"}).json()
    assert r["status"] == "answered" and r["user"] == "EMP-0040"
    assert r["why"][0].startswith("decided by: graph_owner") and r["timing_text"]
    assert any(c["kind"] == "record" for c in r["citations"])
    log = (setup["paths"].log_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(log[-1])
    assert last["channel"] == "api" and last["user"] == "EMP-0040" and last["kind"] == "answer"


def test_ask_is_access_controlled(setup):
    eng = client(setup, "EMP-0040").post("/api/ask", json={"question": INVOICE_Q}).json()
    fin = client(setup, "EMP-0113").post("/api/ask", json={"question": INVOICE_Q}).json()
    assert eng["status"] == "not_found" and fin["status"] == "answered"
    assert not any(h in visible_text(eng, INVOICE_Q) for h in HIDDEN)


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_bad_questions_are_refused(setup, question):
    assert client(setup, "EMP-0040").post("/api/ask", json={"question": question}).status_code == 422


def test_internal_error_is_generic_and_logged(setup, monkeypatch):
    c = client(setup, "EMP-0040")
    monkeypatch.setattr(setup["kora"].answerer, "answer",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret detail PROC-FIN-003")))
    r = TestClient(setup["app"], raise_server_exceptions=False)
    r.cookies = c.cookies
    resp = r.post("/api/ask", json={"question": "anything"})
    assert resp.status_code == 500 and "secret" not in resp.text and "logged" in resp.json()["detail"]
    last = json.loads((setup["paths"].log_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["kind"] == "error" and last["channel"] == "api"


# ---------------- sources ----------------

def _chunk_of(setup, document_id):
    return next(cid for cid, c in setup["kora"].answerer._chunks.items() if c.document_id == document_id)


def test_cited_chunk_and_document_open(setup):
    c = client(setup, "EMP-0040")
    chunk = c.get(f"/api/chunks/{_chunk_of(setup, 'PROC-HR-002')}").json()
    assert chunk["document_id"] == "PROC-HR-002" and "28 days" in json.dumps(chunk)
    doc = c.get("/api/documents/PROC-HR-002").json()
    assert doc["versions"][0]["file"] == "leave_policy.md" and doc["chunks"]


def test_hidden_source_looks_exactly_like_a_missing_one(setup):
    c = client(setup, "EMP-0040")
    hidden_chunk = c.get(f"/api/chunks/{_chunk_of(setup, 'PROC-FIN-003')}")
    missing_chunk = c.get("/api/chunks/NO-SUCH_v1.0_000")
    assert hidden_chunk.status_code == missing_chunk.status_code == 404
    assert hidden_chunk.json() == missing_chunk.json()
    assert c.get("/api/documents/PROC-FIN-003").json() == c.get("/api/documents/NO-SUCH").json()
    assert client(setup, "EMP-0113").get("/api/documents/PROC-FIN-003").status_code == 200


def test_record_and_header_citations(setup):
    c = client(setup, "EMP-0040")
    rec = c.get("/api/citations", params={"id": "record:processes.csv#process_id=PROC-FIN-002#owner_role"}).json()
    assert rec["row"]["owner_role"].startswith("Finance Manager")
    hdr = c.get("/api/citations", params={"id": "header:incident_management.md#Related Processes"}).json()
    assert hdr["document_id"] == "PROC-SUP-002" and any("Related Processes" in l for l in hdr["header"])
    hidden = c.get("/api/citations", params={"id": "record:processes.csv#process_id=PROC-FIN-003#owner_role"})
    assert hidden.status_code == 404


def test_colleague_record_shows_directory_fields_only(setup):
    rid = "record:employees.csv#employee_id=EMP-0085#role_id"
    colleague = client(setup, "EMP-0040").get("/api/citations", params={"id": rid}).json()["row"]
    herself = client(setup, "EMP-0085").get("/api/citations", params={"id": rid}).json()["row"]
    assert "international_relocation_germany" not in colleague and "role_id" in colleague
    assert herself["international_relocation_germany"] == "Yes"


# ---------------- graph ----------------

def test_graph_search_is_access_controlled(setup):
    eng = client(setup, "EMP-0040").get("/api/graph/search", params={"q": "invoice"}).json()
    fin = client(setup, "EMP-0113").get("/api/graph/search", params={"q": "invoice"}).json()
    assert not any("FIN-003" in h["key"] for h in eng)
    assert any(h["key"] == "process:PROC-FIN-003" for h in fin)


def test_graph_entity_with_evidence_per_link(setup):
    r = client(setup, "EMP-0040").get("/api/graph/entity/process:PROC-IMPL-001").json()
    assert r["entity"]["name"] == "Customer Onboarding"
    assert r["links"] and all(l["citation"] and l["tier_label"] for l in r["links"])
    assert any(l["relationship"] == "approves" and l["documented"] for l in r["links"])
    assert client(setup, "EMP-0040").get("/api/graph/entity/process:PROC-FIN-003").status_code == 404


def test_graph_person_view_respects_privacy(setup):
    r = client(setup, "EMP-0040").get("/api/graph/entity/employee:EMP-0085").json()
    assert "international_relocation_germany" not in r["entity"]["attributes"]
    assert not any(l["relationship"] == "eligible_for" for l in r["links"])


# ---------------- admin ----------------

def test_admin_endpoints_are_for_admins_only(setup):
    assert client(setup, "EMP-0040").get("/api/admin/config").status_code == 403
    cfg = client(setup, "EMP-0120").get("/api/admin/config").json()
    assert "access" in cfg and "admins" in cfg


def test_admin_graph_quality_is_their_own_view(setup):
    r = client(setup, "EMP-0120").get("/api/admin/graph-quality").json()
    assert r["links"] > 1000 and "PROC-FIN-003" not in json.dumps(r)          # admins get no extra documents
