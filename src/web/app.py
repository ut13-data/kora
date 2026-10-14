"""
KORA's web API.

DEVELOPMENT LOGIN, NOT PRODUCTION IDENTITY
------------------------------------------
You sign in by choosing an employee from the employees table and typing the
shared development passcode (environment variable KORA_DEV_PASSCODE, default
"kora-dev"). That is enough to TEST employees of every level, and nothing more:
anyone who knows the passcode can be anyone. A real deployment replaces
/api/login with the company's single sign-on and keeps everything else.

What the login is trusted for, and what it is not:
  * The browser only holds a SIGNED session cookie containing an employee ID.
    The signature (KORA_SECRET; a random one per run if unset) means it can't
    be edited into another ID without the secret.
  * Groups, department, admin rights: always looked up on the server from the
    employees table and the config, never taken from the browser.
  * Every endpoint that shows company knowledge requires a login, and passes
    that user to Kora.ask / the graph view / the access policy. There is no
    "no user" (full access) path in the API.

Endpoints (all JSON; interactive docs at /docs):
  GET  /api/health                 index status, dev-login warning
  GET  /api/login/employees        the development picker (directory fields only)
  POST /api/login                  {employee_id, passcode}
  POST /api/logout
  GET  /api/me                     who is signed in (+ admin flag)
  POST /api/ask                    {question} -> answer, citations, why, timings
  GET  /api/chunks/{chunk_id}      a cited passage, with its document's details
  GET  /api/documents/{doc_id}     a whole document (all its passages)
  GET  /api/citations?id=...       a header-line or company-record citation
  GET  /api/graph/search?q=...     find entities by name
  GET  /api/graph/entity/{key}     one entity and its links, each with evidence
  GET  /api/me/context             "My week": your own role, processes, approvals, Germany plan, tone
  GET  /api/admin/config           the company config (admins only)
  GET  /api/admin/config/schema    threshold ranges and group syntax, for the settings forms
  POST /api/admin/config/validate  check settings without saving
  PUT  /api/admin/config           validate, keep a backup, save, reload KORA
  GET  /api/admin/config/backups   earlier versions;  POST /api/admin/config/restore {name}
  GET  /api/admin/graph-quality    graph health as this admin may see it (admins only)

Pages (M5, plain HTML + small scripts on top of these endpoints, no build step):
  /login  /ask  /explore  /me  /admin  /documents/{id}

A hidden document answers exactly like a missing one (404 "not found"), so the
API can't be used to discover what exists.
"""

import json
import os
import secrets
import threading
from pathlib import Path

from typing import Literal

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from src.graph.service import _RECORD_ID as RECORD_ID
from src.index import Kora, Paths

DEV_PASSCODE_ENV, SECRET_ENV = "KORA_DEV_PASSCODE", "KORA_SECRET"
DEFAULT_DEV_PASSCODE = "kora-dev"
MAX_QUESTION = 500
NOT_FOUND = "Not found."                 # the same words for missing and for hidden


class LoginRequest(BaseModel):
    employee_id: str = Field(min_length=1, max_length=40)
    passcode: str = Field(min_length=1, max_length=200)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    channel: Literal["ui", "api"] = "api"            # the KORA screens send "ui" (for the dashboards)


class RestoreRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


WEB_DIR = Path(__file__).parent


# ---------------- serialising answers ----------------

def answer_json(answer) -> dict:
    from src.retrieval.answer import explain_decision, format_timings
    return {
        "question": answer.question,
        "status": answer.status,
        "lines": [{"kind": l.kind, "text": l.text, "detail": l.detail, "citation": l.citation,
                   "chunk_id": l.chunk_id} for l in answer.lines],
        "citations": [{"number": c.number, "kind": c.kind, "chunk_id": c.chunk_id, "document_id": c.document_id,
                       "title": c.title, "version": c.version, "status": c.status,
                       "effective_date": c.effective_date, "section": c.section, "file": c.source_filename,
                       "excerpt": c.excerpt, "label": c.label()} for c in answer.citations],
        "notes": answer.notes,
        "uncovered_terms": answer.uncovered_terms,
        "why": explain_decision(answer.decision) if answer.decision else [],
        "decision": answer.decision,
        "timings": answer.timings,
        "timing_text": format_timings(answer.timings, answer.startup) if answer.timings else "",
        "startup": answer.startup,
        "user": answer.user,
    }


def entity_json(e) -> dict:
    return {"key": e.key, "id": e.entity_id, "type": e.entity_type, "name": e.name, "aliases": e.aliases,
            "attributes": e.attributes, "source_documents": e.source_documents}


# ---------------- the app ----------------

def create_app(paths: Paths | None = None, secret: str | None = None, dev_passcode: str | None = None,
               kora: Kora | None = None) -> FastAPI:
    secret = secret or os.environ.get(SECRET_ENV) or secrets.token_hex(32)
    dev_passcode = dev_passcode or os.environ.get(DEV_PASSCODE_ENV) or DEFAULT_DEV_PASSCODE
    app = FastAPI(title="KORA", version="5.0",
                  description="KORA knowledge API. DEVELOPMENT LOGIN: not production identity (see /api/health).")
    app.add_middleware(SessionMiddleware, secret_key=secret, session_cookie="kora_session",
                       max_age=8 * 3600, same_site="lax", https_only=False)
    state = {"kora": kora or Kora.open(paths or Paths(), channel="api")}
    lock = threading.RLock()                 # Kora rebuilds and caches in place: one request at a time

    def core() -> Kora:
        return state["kora"]

    # ---- who is asking ----

    def current_user(request: Request):
        emp = request.session.get("employee_id")
        if not emp:
            raise HTTPException(401, "Please sign in.")
        with lock:
            user = core().user(emp)
        if user is None:                     # removed from the employees table since signing in
            request.session.clear()
            raise HTTPException(401, "Please sign in again.")
        return user

    def admin_user(user=Depends(current_user)):
        if not core().answerer.config.is_admin(user):
            raise HTTPException(403, "Admins only.")
        return user

    def can_read(user, document_id: str) -> bool:
        return core().answerer.access.can_read(user, document_id)

    # ---- health and login ----

    @app.get("/api/health")
    def health():
        with lock:
            status = core().status()
        return {"index_current": status.current, "index": status.summary(),
                "login": "development: choose an employee + shared passcode. NOT production identity.",
                "graph": core().answerer.graph is not None, "events": core().events[-5:]}

    @app.get("/api/login/employees")
    def login_employees():
        """The development picker: name, role, department, location -- the directory, nothing private."""
        with lock:
            core().refresh()
            tables = core().answerer.retriever.tables
        rows = tables.get("employees") or []
        depts = {d.get("dept_id"): d.get("name") for d in tables.get("departments") or []}
        cities = {l.get("location_id"): l.get("city") for l in tables.get("locations") or []}
        return [{"employee_id": r.get("employee_id"), "name": f"{r.get('first_name', '')} {r.get('last_name', '')}".strip(),
                 "role": r.get("role_title"), "department": depts.get(r.get("dept_id")) or r.get("dept_id"),
                 "location": cities.get(r.get("location_id")) or r.get("location_id")}
                for r in rows if r.get("employee_id")]

    @app.post("/api/login")
    def login(body: LoginRequest, request: Request):
        if not secrets.compare_digest(body.passcode, dev_passcode):
            raise HTTPException(401, "Employee or passcode not recognised.")
        with lock:
            user = core().user(body.employee_id.strip())
        if user is None:
            raise HTTPException(401, "Employee or passcode not recognised.")     # same words: no ID probing
        request.session.clear()
        request.session["employee_id"] = user.user_id
        return me(user)

    @app.post("/api/logout")
    def logout(request: Request):
        request.session.clear()
        return {"signed_out": True}

    @app.get("/api/me")
    def me(user=Depends(current_user)):
        return {"employee_id": user.user_id, "name": user.name, "groups": sorted(user.groups),
                "admin": core().answerer.config.is_admin(user)}

    # ---- asking ----

    @app.post("/api/ask")
    def ask(body: AskRequest, user=Depends(current_user)):
        question = " ".join(body.question.split())
        if not question:
            raise HTTPException(422, "Please type a question.")
        with lock:
            answer = core().ask(question, user=user, channel=body.channel)
        return answer_json(answer)

    # ---- sources ----

    @app.get("/api/chunks/{chunk_id}")
    def chunk(chunk_id: str, user=Depends(current_user)):
        with lock:
            core().refresh()
            c = core().answerer._chunks.get(chunk_id)
            if c is None or not can_read(user, c.document_id):
                raise HTTPException(404, NOT_FOUND)
            meta = core().answerer.retriever.documents.get((c.document_id, c.version))
        return {"chunk_id": c.chunk_id, "document_id": c.document_id, "title": c.document_title,
                "version": c.version, "status": c.status, "section": c.section, "text": c.chunk_text,
                "file": getattr(meta, "source_filename", None), "effective_date": getattr(meta, "effective_date", None)}

    @app.get("/api/documents/{document_id}")
    def document(document_id: str, user=Depends(current_user)):
        with lock:
            core().refresh()
            if not can_read(user, document_id):
                raise HTTPException(404, NOT_FOUND)
            chunks = sorted((c for c in core().answerer._chunks.values() if c.document_id == document_id),
                            key=lambda c: (c.version, c.chunk_index))
            if not chunks:
                raise HTTPException(404, NOT_FOUND)
            metas = [m for (d, _), m in core().answerer.retriever.documents.items() if d == document_id]
        return {"document_id": document_id,
                "versions": [{"version": m.version, "title": m.title, "status": m.status,
                              "effective_date": m.effective_date, "file": m.source_filename,
                              "owner": m.owner_role, "department": m.department} for m in metas],
                "chunks": [{"chunk_id": c.chunk_id, "version": c.version, "section": c.section,
                            "text": c.chunk_text} for c in chunks]}

    @app.get("/api/citations")
    def citation(id: str, user=Depends(current_user)):
        """A header-line or company-record citation, checked again now."""
        with lock:
            core().refresh()
            graph = core().answerer.graph
            if graph is None or not id.startswith(("header:", "record:", "file:")):
                raise HTTPException(404, NOT_FOUND)
            d = graph.describe(id)
            if d["kind"] in ("header", "file"):
                meta = next((m for m in core().answerer.retriever.documents.values()
                             if m.source_filename.lower() == d["file"].lower()), None)
                if meta is None or not can_read(user, meta.document_id):
                    raise HTTPException(404, NOT_FOUND)
                raw = graph.tracer.raw_text(d["file"]) or ""
                lines = [l.strip() for l in raw.splitlines() if l.strip().startswith("**") and ":**" in l][:12]
                return {"kind": d["kind"], "document_id": meta.document_id, "title": meta.title,
                        "file": d["file"], "section": d["section"], "header": lines}
            hidden = {doc for doc in core().answerer._all_documents if not can_read(user, doc)}
            if any(h in id for h in hidden):
                raise HTTPException(404, NOT_FOUND)
            m = RECORD_ID.match(id)
            rows = (graph.tracer.tables.get(Path(m["file"]).stem) or []) if m else []
            row = next((r for r in rows if str(r.get(m["kcol"])) == m["key"]), None) if m else None
            if row is None:
                raise HTTPException(404, NOT_FOUND)
            if m["file"] == "employees.csv" and row.get("employee_id") != user.user_id \
                    and not core().answerer.config.is_admin(user):
                visible = set(core().answerer.config.colleague_fields) | {"employee_id", "first_name", "last_name"}
                row = {k: v for k, v in row.items() if k in visible}
            return {"kind": "record", "file": m["file"], "key_column": m["kcol"], "key": m["key"],
                    "column": m["col"], "row": row}

    # ---- graph ----

    def queries(user):
        graph = core().answerer.graph
        if graph is None:
            raise HTTPException(404, "The knowledge graph is switched off in the config.")
        return graph.queries(user)

    @app.get("/api/graph/search")
    def graph_search(q: str, user=Depends(current_user)):
        q = q.strip().lower()
        if len(q) < 2:
            return []
        with lock:
            core().refresh()
            view = queries(user).view
            hits = [e for e in view.entities() if q in e.name.lower() or q == e.entity_id.lower()
                    or any(q == a.lower() for a in e.aliases)]
        hits.sort(key=lambda e: (not e.name.lower().startswith(q), e.entity_type, e.name))
        return [{"key": e.key, "name": e.name, "type": e.entity_type} for e in hits[:25]]

    @app.get("/api/graph/entity/{key:path}")
    def graph_entity(key: str, user=Depends(current_user)):
        from src.graph.schema import DOCUMENTED, TIER_LABELS
        from src.intelligence.graph import BOTH
        with lock:
            core().refresh()
            q = queries(user)
            node = q.view.node(key)
            if node is None:
                raise HTTPException(404, NOT_FOUND)
            links = []
            for hop in q.view.hops(key, None, BOTH):
                fact = q.fact(hop.relationship)
                if fact is None:                       # nothing verifies: not shown
                    continue
                other = q.view.node(hop.node)
                links.append({"relationship": hop.relationship.relationship, "direction": hop.direction,
                              "other": {"key": hop.node, "name": other.name if other else hop.node,
                                        "type": hop.node.split(":", 1)[0]},
                              "claim": fact.claim, "tier": fact.tier, "tier_label": TIER_LABELS[fact.tier],
                              "documented": fact.tier in DOCUMENTED, "historical": fact.historical,
                              "citation": fact.citation, "relationship_id": hop.relationship.relationship_id})
        links.sort(key=lambda l: (not l["documented"], l["relationship"], l["other"]["name"]))
        return {"entity": entity_json(node), "links": links}

    # ---- admin ----

    @app.get("/api/admin/config")
    def admin_config(user=Depends(admin_user)):
        path = core().paths.config
        return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}

    @app.get("/api/admin/graph-quality")
    def admin_graph_quality(user=Depends(admin_user)):
        from src.graph.quality import quality_report
        with lock:
            core().refresh()
            q = queries(user)
            report = quality_report(q.view, q.tracer, core().paths.intelligence_dir,
                                    can_read=lambda d: can_read(user, d))
        return report

    # ---- my week ----

    def hidden_names(user) -> set[str]:
        """IDs and names of documents, processes and topics this user may not see."""
        graph = core().answerer.graph
        if graph is None:
            return {d for d in core().answerer._all_documents if not can_read(user, d)}
        view = graph.queries(user).view
        return {x for k, e in graph.graph.index.entities.items()
                if e.entity_type in ("document", "process", "policy", "topic") and not view.visible(k)
                for x in (e.entity_id, e.name)}

    @app.get("/api/me/context")
    def me_context(user=Depends(current_user)):
        from src.web.context import load_context, my_context
        with lock:
            core().refresh()
            data = load_context(core().paths.intelligence_dir)
            ctx = my_context(data, user.user_id, lambda d: can_read(user, d), hidden_names(user))
        if ctx is None:
            raise HTTPException(404, NOT_FOUND)
        return ctx

    # ---- settings (admins) ----

    def employee_ids() -> set[str]:
        return set(core().answerer.retriever.users)

    @app.get("/api/admin/config/schema")
    def admin_schema(user=Depends(admin_user)):
        from src.web.settings import schema
        rows = core().answerer.retriever.tables.get("employees") or [{}]
        return schema(list(rows[0].keys()))

    @app.post("/api/admin/config/validate")
    def admin_validate(raw: dict = Body(...), user=Depends(admin_user)):
        from src.web.settings import check
        with lock:
            problems = check(raw, employee_ids(), user.user_id)
        return {"valid": not problems, "problems": problems}

    @app.put("/api/admin/config")
    def admin_save(raw: dict = Body(...), user=Depends(admin_user)):
        from src.web.settings import check, save
        with lock:
            problems = check(raw, employee_ids(), user.user_id)
            if problems:
                return JSONResponse({"saved": False, "problems": problems}, status_code=422)
            backup = save(core().paths.config, raw)
            core().reload()
        return {"saved": True, "backup": backup, "warnings": core().answerer.config.warnings}

    @app.get("/api/admin/config/backups")
    def admin_backups(user=Depends(admin_user)):
        from src.web.settings import backups
        return backups(core().paths.config)

    @app.post("/api/admin/config/restore")
    def admin_restore(body: RestoreRequest, user=Depends(admin_user)):
        from src.web.settings import check, read_backup, save
        with lock:
            try:
                raw = read_backup(core().paths.config, body.name)
            except FileNotFoundError:
                raise HTTPException(404, NOT_FOUND)
            problems = check(raw, employee_ids(), user.user_id)
            if problems:
                return JSONResponse({"saved": False, "problems": problems}, status_code=422)
            backup = save(core().paths.config, raw)
            core().reload()
        return {"saved": True, "restored": body.name, "backup": backup}

    # ---- pages ----

    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    def page(request: Request, name: str, title: str, **extra):
        emp = request.session.get("employee_id")
        with lock:
            user = core().user(emp) if emp else None
        if user is None:
            return RedirectResponse("/login", status_code=303)
        admin = core().answerer.config.is_admin(user)
        if name == "admin.html" and not admin:
            return RedirectResponse("/ask", status_code=303)
        return templates.TemplateResponse(request, name, {"title": title, "user": user, "admin": admin,
                                                          "first_name": user.name.split(" ")[0], **extra})

    @app.get("/", include_in_schema=False)
    def home(request: Request):
        return RedirectResponse("/ask" if request.session.get("employee_id") else "/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html", {"title": "Sign in"})

    @app.get("/ask", response_class=HTMLResponse, include_in_schema=False)
    def ask_page(request: Request):
        return page(request, "ask.html", "Ask")

    @app.get("/explore", response_class=HTMLResponse, include_in_schema=False)
    def explore_page(request: Request):
        return page(request, "explore.html", "Explore")

    @app.get("/me", response_class=HTMLResponse, include_in_schema=False)
    def me_page(request: Request):
        return page(request, "me.html", "My week")

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_page(request: Request):
        return page(request, "admin.html", "Settings")

    @app.get("/documents/{document_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_page(request: Request, document_id: str):
        return page(request, "document.html", "Document", document_id=document_id)

    @app.exception_handler(Exception)
    def unexpected(request: Request, exc: Exception):    # details are in the log, not in the browser
        return JSONResponse({"detail": "KORA hit an internal error. It has been logged."}, status_code=500)

    return app
