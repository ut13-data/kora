# KORA Phase 5 M3: graph answers in `Kora.ask`, findings

_Written by Claude on 2026-10-14. Numbers come from `python -m src.evaluation.baseline` and
`data/evaluation/results_graph_*.json`._

## What changed in the answer flow

```
Question -> Phase 4 (route, retrieve, compose)              unchanged, always runs
         -> graph (per-user view): intent + visible entity + documented links?
              no  -> Phase 4's answer is the answer
              yes -> graph lines (claim + verbatim evidence), verified again at answer time
                     Phase 4 found a CONFLICT -> conflict sides kept first, status stays "conflict"
                     no graph line survives verification -> Phase 4's answer stands
```

- New answer line kinds: `relation` (a documented link) and `possible` (only cross-referenced or mentioned together).
- New citation kinds: `header` (a document's header line) and `record` (a company table row and column), plus a raw-file quote for the org chart.
- Every graph line is checked again at answer time, from its citation ID and text alone: the header line must still be in the file, the record value still in the table, the quote still in the chunk. Evaluation uses the same check (`faithful`).
- `decision` records the route (`graph_<kind>`), what the graph found, and what Phase 4 alone would have said (`--why`). There's a new timing step, `graph_ms`, and a start-up step, `graph`.
- Config: `"graph": {"enabled": true}`. Without a config the graph is off, like every other feature.
- Citation precision in the evaluation ignores company-record citations, because they aren't documents.

## Results against the frozen Phase 4 baseline

| Set | n | Status | Expected names | Key fact | No leak | Faithful |
|---|---|---|---|---|---|---|
| Phase 4 development | 41 | 0.98 -> 0.98 | n/a | 1.00 -> 1.00 | 1.00 | 1.00 |
| Phase 4 held-out v1-v5 | 50 | unchanged (every question) | n/a | unchanged | 1.00 | 1.00 |
| Graph development (used while building) | 22 | 0.59 -> **1.00** | 0.12 -> **1.00** | 0.33 -> 1.00 | 1.00 | 1.00 |
| Graph held-out, all | 12 | 0.75 -> **1.00** | 0.19 -> **1.00** | 1.00 | 1.00 | 1.00 |
| Graph held-out, unseen only | 9 | 9/9 | 1.00 | n/a | 1.00 | 1.00 |
| Graph held-out, seen by mistake (H02, H04, H09) | 3 | 3/3 | 1.00 | 1.00 | 1.00 | 1.00 |

No Phase 4 question changed status or citations: BROKEN 0. Twelve graph questions went from wrong to right. Thirteen kept a correct status but now answer with the expected names; before, most "answered" with a sentence that didn't name anyone.

Latency per answer: p50 ~6 ms, p95 ~9 ms (unchanged). Start-up +0.04 s for the graph.

## How the held-out run happened (honest account)

- The held-out set was run once, when `python -m src.evaluation.baseline` was first run after integration. That was a little before the tests and docs for M3 were finished, but after all routing and answer code. That run is the result above.
- After it, three changes were made. All were general, none targeted a held-out question, and the held-out results were re-checked as identical:
  1. "X **has a step** in P" became "X **is named in a step of** P". Phase 3 links every role named in a step, including steps done by someone else ("Consultant reviews the AE's handoff record"). Visible in development G01 too.
  2. Tracer hole: when a CSV value no longer matched a link, the check fell back to "derived from the row" instead of "unverified". Now only links computed from several columns (relocation eligibility) can be "derived"; a single-column mismatch is unverified. This was found by a test that changes a record after loading. Users never saw these lines, because derived evidence is never shown.
  3. "Possibly related" lines now only say "X and Y are mentioned together", never a stronger verb.
- H02, H04 and H09 were seen during M2 (see their notes in the question file).
- Both graph question sets were written by the builder (Claude), who also designed the intents. 9/9 on 12 builder-written questions shows the design works for the question types it was designed for. It doesn't prove KORA handles new wording. UT's own questions in the UI are the real test.

## Known limits

- **The graph only knows the question types in `src/graph/query.py`**: involved, owner, policies, impact, delay, next, department, role, person, uses, which department, head of the owning department, and old-version approvals. Anything else is Phase 4.
- **Entity names must appear in the question.** "bug reports" does not find "Bug Reporting and Escalation" (H04 was answered by Phase 4, partial). Unfamiliar everyday wording is still the main weakness, as in Phase 4.
- **Approvals stay with Phase 4** on purpose: its rule engine knows amounts and tiers.
- **"Why delayed" can only list documented dependencies and known gaps.** KORA has no live status data and says so in every such answer.
- **"What's affected" lists documented dependencies (quotes) and cross-references (headers).** No document describes the actual effect of a change, and KORA says so.
- **Record citations show IDs** (`owning_dept_id = DEPT-PROD`). The claim above them uses names; the UI can show both.
- **Colleague privacy is a provisional config** (`person_visibility`), to be decided with test employees in the UI.

## Questions to try in the UI (graph)

Answered: Which teams are involved in customer onboarding? | Who owns the procurement process? | Which processes use NexCRM? | Who heads the department that owns NexFlow? | Which processes is the Account Executive part of? | What does Lena Hoffmann do? | Which process starts when a deal is closed?
Partial (by design): Why is the customer onboarding process delayed? | If the Data Security Policy changes, which processes might be affected? | What comes after lead qualification?
History: Who gave final approval for customer onboarding before 2025?
Access: "Who takes part in the invoice and payment process?" as EMP-0040 (not found) vs EMP-0113 (answered).
Privacy: "What does Julia Zimmermann do?" as EMP-0040 (directory facts only) vs as EMP-0085 herself.
