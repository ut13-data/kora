# KORA Phase 5 handoff (to Phase 6)

Written 2026-10-15 at the end of Phase 5 (M7). Read this first when starting Phase 6. The plan and decisions behind Phase 5 are in the KORA project doc `claude/phase5_plan.md`; the numbers below come from `reports/phase5_final_evaluation.md`.

## 1. What Phase 5 delivered

| Part | Where | What it does |
|---|---|---|
| Graph KORA can trust | `src/graph/` | Reads the Phase 3 graph and gives every link an **evidence tier**: *quote* (sentence found in its chunk), *header* (a `**Field:** value` line), *record* (a CSV cell), *mention*, *derived*. Only quote, header and record count as documented and may be stated as fact. |
| Per-user graph | `src/graph/view.py` | A hidden document also hides its processes, topics, links, evidence and counts. Colleagues see only `person_visibility.colleague_fields`; the person and admins see all fields. Admins get **no** extra document access. |
| Relationship answers | `src/graph/resolve.py`, `query.py`, `service.py` | Owner, involved, next step, uses, policies, department, head of owner, impact, "why delayed" (always partial: no live status), approval history. The verb follows the evidence ("lists X as related", never "governs"). The graph stays silent unless there is an intent, a visible entity and a documented link; then Phase 4 answers as before. |
| One answer path | `src/retrieval/answer.py` | `Kora.ask` runs Phase 4, then the graph. Header and record citations use pseudo IDs (`header:...`, `record:...`) and are re-checked against the files at answer time. Phase 4 conflicts are kept first. |
| Quieter answers | `answer.py` (M4) | A sentence about another office is dropped unless the question asked about that office. Relevant two-line answers stay. |
| Web API and screens | `src/web/` | `python -m src.web` (127.0.0.1:8000). Sign in, Ask (sources panel, "Why this answer"), Explore (graph), My week (checklist + communication guidance), Document, Settings (validated, backups, restore), Dashboards. FastAPI + HTML + light JS, no build step. |
| Dashboards | `src/analytics/answers_log.py` | From the real `data/logs/answers.jsonl`: answer status, errors kept apart from not_found, unknown words, refused/partial questions, weak topics, speed, knowledge quality, usage. Totals only. |
| Measurement | `src/evaluation/baseline.py` | Frozen Phase 4 results; every change is labelled FIXED / BROKEN / CHANGED; a new leak always counts as BROKEN. |

## 2. What was tested, with results

* **498 tests pass** (2 skipped: the optional embedding model). Run: `python -m pytest test/ -q`.
* **Phase 4 is unchanged**: all 91 questions in the Phase 4 sets give the same correctness as before; 0 questions got worse.
* **Relationship questions** (written for Phase 5):

| Set | Correct status | Expected names stated | Leaks | Faithful |
|---|---|---|---|---|
| Development (22) | 0.59 -> 1.00 | 0.12 -> 1.00 | 0 | 1.00 |
| Held-out (12, 9 unseen) | 0.75 -> 1.00 | 0.19 -> 1.00 | 0 | 1.00 |

* **Graph health**: 63% of links are documented; all 1,974 evidence items still check out against the source files (`reports/graph_quality.md`).
* **Speed**: about 6 ms median per answer on the demo data.
* **Browser check** (M7): documented, partly documented, sources disagree and not found rendered in light and dark mode and at phone width (390 px, no sideways scroll). Questions typed as HTML are shown as text.

How much to trust these numbers: the question sets were written by the builder (Claude), and three held-out questions (H02, H04, H09) were seen while building. They show the design works as intended on Nexora; they are not an independent test. UT's own questions in the Ask screen are the real check.

## 3. What is incomplete or limited

1. **Everyday wording.** "bug reports" doesn't find "Bug Reporting and Escalation". Today the fix is the word list in Settings.
2. **Stray lines from Phase 4 ranking.** Example: the security-training conflict also shows an example sentence from the communication guide. The graph doesn't touch these answers.
3. **Answers sound mechanical**: quotes plus short connecting sentences, no conversation, no follow-ups.
4. **Record citations show IDs** (`row PROC-FIN-002, column owner_role`), not friendly names.
5. **My week is read-only.** The checklist can't be ticked from the UI; progress lives in `data/context/`.
6. **Demo date.** My week uses `as_of` 2025-04-22 so the Nexora data looks current.
7. **No live status.** KORA knows documents, not what is happening in a case today, so "why is X delayed" is always partial.
8. **Single process, local only.** One lock serialises requests; fine for a demo, not for many users.

## 4. Open decisions (UT)

| Decision | Today | When |
|---|---|---|
| What colleagues see about each other | Provisional: role, role title, department, location, manager (`person_visibility.colleague_fields`) | After UT tests employees of every level in the UI |
| Login | Development only: pick an employee + shared passcode (`KORA_DEV_PASSCODE`, default `kora-dev`) | Before anyone but UT uses it: company sign-in (SSO) |
| Demo date `as_of` | 2025-04-22 | When real data replaces Nexora |
| Checklist editing | Read-only | Phase 6 (memory of progress) |
| Admins | EMP-0120 (Head of HR), EMP-0126 (CISO) | Any time, in Settings |
| Dashboard question text | Shown when at least 2 different people asked it, never when it names a person | Any time, in Settings |

## 5. Recommended starting point for Phase 6

Keep KORA's core rule: **a fact is only stated if a quote, header line or record backs it, and that is checked by code, not by the model.** Build the agent around it:

1. **Language model for wording and conversation.** The model receives KORA's verified answer lines and citations and writes a natural reply, asks a clarifying question when the question is vague, and handles follow-ups ("and who approves that?"). Afterwards, KORA checks every factual sentence against the cited evidence (`verify_line` and `GraphService.verify` exist today); a sentence that fails is dropped or flagged. Measure with the same baseline tool: faithfulness must stay 1.00 and no new leaks.
2. **Wording help.** Let the model suggest synonyms for unknown words (the dashboard already lists them "to review"); an admin approves them into the word list. This attacks limit 1 without letting the model invent facts.
3. **Memory.** Per-employee progress (checklist items done, questions asked, preferences), stored by KORA with the same access rules. Makes My week editable.
4. **Planning and proactive check-ins** for the relocation buddy: the Germany checklist and communication guidance are already structured (`data/intelligence/germany_checklists.json`, `communication_guidance.json`); the agent turns them into a week plan and reminders. Phase 5 deliberately has no proactive behaviour.
5. **Website Intelligence Agent**: a separate source with its own provenance tier (web page + fetch date), never mixed silently with company documents.

Before starting: decide the model and where it runs (data leaves the machine or not), keep the "no LLM" path as a fallback, and add a Phase 6 baseline freeze (`python -m src.evaluation.baseline --freeze --baseline data/evaluation/baseline_phase5.json`).

## 6. Questions to try in the UI

Sign in with different employees to see access and privacy work:

| Employee | Who | Shows |
|---|---|---|
| EMP-0040 | Senior Software Engineer, Bangalore | Can't see the Invoice and Payment Process (Finance/Operations only) |
| EMP-0113 | Finance Manager, Munich | Can see it |
| EMP-0085 | Implementation Consultant, Munich, relocated to Germany | My week with the Germany checklist |
| EMP-0120 | Head of HR, admin | Settings and Dashboards |

Questions:

* Who owns the procurement process? (documented, from company records)
* Which teams are involved in customer onboarding?
* Why is the customer onboarding process delayed? (partial: no live status, with known gaps)
* If the Data Security Policy changes, which processes might be affected?
* Who takes part in the invoice and payment process? (try as EMP-0040, then EMP-0113)
* How long do new hires have to complete security training? (sources disagree)
* Who gave final approval for customer onboarding before 2025?
* What is the company policy on bringing pets to the office? (not found)
* How many days of annual leave do employees in Germany get? (no US lines)

## 7. Run it

```
python -m pytest test/ -q
python -m src.evaluation.baseline --report reports/phase5_final_evaluation.md
python -m src.graph.quality
python -m src.web            # http://127.0.0.1:8000, passcode kora-dev
```
