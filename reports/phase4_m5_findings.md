# KORA Phase 4 - M5 evaluation findings (hand-written summary)

Generated reports with every number: `evaluation_results.md` (41 development questions),
`evaluation_heldout_v1.md` / `evaluation_heldout.md`, `evaluation_heldout_v2_untouched.md` /
`evaluation_heldout_v2.md`. Raw data: `data/evaluation/results*.json`.

## Method

* 41 development questions across all 11 brief categories (14 seen while building M1-M4, 27 new).
* Held-out v1 (10): written before the M5 fixes, run once, then used for diagnosis.
* Held-out v2 (10): written after v1's diagnosis and before the fixes it led to, run once
  untouched (`results_heldout_v2_untouched.json`), then once more after the final abstention fix.
* Gold labels are document IDs; every label was checked against the documents before scoring.
* Same author built KORA and wrote all questions (agreed with UT); the untouched held-out
  numbers are the fairest estimate.

## Results

| Set | Plain TF-IDF: right doc in top 5 | KORA: right doc | Correct status | Key fact | Faithful |
|---|---|---|---|---|---|
| Development, before M5 fixes | 0.90 | 0.95 | 0.66 | 0.76 | 0.98 |
| Development, final | 0.90 | 1.00 | 0.88 | 0.90 | 1.00 |
| Held-out v1, untouched | 1.00 | 0.89 | 0.70 | 0.50 | 1.00 |
| Held-out v2, untouched | 0.88 | 1.00 | 0.50 | 0.88 | 1.00 |

Latency: full answer (route + retrieve + verify) p50 ~6 ms, p95 ~9 ms. Load 0.4 s, ~6 MB.
Full rebuild after new files: ~1-3 s for this corpus.

## What the numbers say

* The ASRS-style routing helps retrieval on unseen questions (v2: 1.00 vs 0.88), at almost no cost.
  On v1 it lost one question to plain search (router sent "legacy NexFlow" to the product page);
  fixed generally: the safety-net results may now be quoted when they beat the router's choice.
* Nothing invented: faithfulness 1.00 on every set, enforced at runtime (unverifiable lines are dropped).
* After the final abstention rule, no question in either held-out set was answered when it should
  have been refused, or refused when answerable.
* Status labels are the weak spot on unseen questions: correct answers are often marked "partial"
  because the question uses a word the source doesn't ("how deep ... nested" vs "two levels of
  nesting"). Safe, but noisy.

## Known limitations (honest)

1. Lexical matching: synonyms ("money back" vs "reimbursed", "pushed live" vs "deploys") make KORA
   abstain (safe) instead of answering. Planned fix: semantic embeddings (M5b), re-evaluated on a new held-out set.
2. Homonyms: "stock price" matches inventory "stock" (development Q39 still answers "partial").
3. "Partial" over-flagging on paraphrased questions (see above).
4. Every threshold was chosen on this one synthetic company; the held-out sets are only 20 questions.
5. Template answers (quotes, not prose); access control not yet built (M7).

## M5b: semantic embeddings (all-MiniLM-L6-v2), honest result

What was added: meaning-based search fused with word search by rank (RRF), meaning-based quotes,
and paraphrase acceptance for words the knowledge base never uses. Safety rules: every quote is still
verified word-for-word; meaning never excuses a word the knowledge base has elsewhere
("contractors"); names are never matched by meaning ("Paris" is not "Berlin"); if the model can't
load, KORA runs on words only. Thresholds were calibrated on the real model on UT's machine
(`src.evaluation.calibrate_semantic`, development + v1 + v2 only): strong 0.59, quote 0.58, word 0.54.

| Set | Plain TF-IDF: right doc in top 5 / MRR | KORA: right doc / MRR | Correct status | Key fact | Faithful |
|---|---|---|---|---|---|
| Development, words only | 0.90 / 0.70 | 1.00 / 0.90 | 0.88 | 0.90 | 1.00 |
| Development, with embeddings | 0.90 / 0.70 | 1.00 / 0.95 | 0.88 | 0.90 | 1.00 |
| Held-out v3, words only (sandbox) | 1.00 / 0.79 | 1.00 / 0.76 | 0.70 | 0.63 | 1.00 |
| Held-out v3, with embeddings (official) | 1.00 / 0.79 | 1.00 / 0.75 | 0.70 | 0.63 | 1.00 |

What this says:
* Embeddings improved RANKING on the development set (MRR 0.90 -> 0.95), fixed two citation
  mistakes (Q05, Q27) and reduced a third (Q19: 0.33 -> 0.67). Answer statuses were unchanged.
* On the untouched held-out v3 they changed NOTHING in the answers: the same 4 questions failed
  with and without them (W01, W02, W03, W07). KORA's ranking on v3 was slightly below plain TF-IDF
  (0.75 vs 0.79).
* All 4 v3 failures were safe ones (refused, partial, or one extra citation); both no-answer traps
  (company car, annual bonus) were correctly refused. No wrong answers.
* Why no gain: a paraphrase must pass TWO gates -- the single word ("trip" ~ "travel" 0.79 passes 0.54)
  AND the whole answer's meaning match (0.59). The second gate, set above every no-answer question,
  blocks most real paraphrases too. Next step: diagnose with the "why" view on development questions
  only, change the gate if the data supports it, and test once on the new held-out v4 (written before
  that change).
* Latency: a full answer takes ~30-36 ms p50, ~60-70 ms p95 with embeddings (encoding the question).
  An earlier "10 ms" figure was a measuring artefact: the baseline had already encoded the question.
* Start-up: ~38-51 s on UT's Windows machine with embeddings (0.5 s without), now broken down per
  step in every answer and in the evaluation report. KORA loads once per process, so this is paid
  at start, not per question.

Limitations added by M5b:
6. Meaning scores of right and wrong documents overlap heavily (meaning alone picks the right
   document in only 38 of 55 questions), so meaning stays a helper to words, never the decider.
7. The paraphrase word list used for calibration is small (8 + 8 pairs); "money" ~ "expenses" (0.49)
   scores the same as the unrelated "canteen" ~ "facilities" (0.49), so some real paraphrases will
   always be refused at a precision-first threshold.

## M5b decision: simpler is better (embeddings OFF by default)

The "why" view (development questions only) showed the real causes of the remaining misses, and
none of them needed embeddings:
* Q37: the right sentence ("Chat ... default for quick questions") was in a chunk that ranked low
  overall, and chunk-level filtering threw it away. Fix: a low-ranked chunk may be quoted when one
  sentence (with its section heading) covers at least 2 main question words and more than any other.
* Q08: "paying" was not recognised as a form of "pay" (stemming), and "trip"/"money back" are
  company vocabulary. Fix: -ing stemming keeps 3 letters; a short company word list in the config
  ("trip" -> travel, "money back" -> reimbursed, "pushed live" -> deploys/release). The list only
  adds search words; the question, the quotes (word-for-word) and the abstention rules are unchanged,
  and every answer says when the list was used.
* With the real model, "money" was judged close to "business" -- a word of the question itself.
  Guarded (a question word never explains another), but it showed how fragile meaning matching is.

Result on development (words only + word list): correct status 0.88 -> 0.95, key fact 0.90 -> 1.00
(Q08, Q10, Q37 fixed); held-out v1 and v2 unchanged (no regressions). Start-up back to ~0.5 s,
answers ~10 ms, no PyTorch needed. Embeddings stay in the code as an option (config:
embeddings.enabled), with calibration and tests, for a later trial with a question-answer model.

Honesty notes: the word list was written from development failures only. Held-out v3 is no longer a
fair test (its W01 uses "trip"); held-out v4 was written before these changes and its everyday words
(boss, outage, supplier, ...) are deliberately NOT in the list -- it measures the system as shipped.

## Held-out v4 (official, run once on the simplified system) and what followed

Official v4 result (words + company word list, embeddings off): correct status 0.30, key fact 0.14,
right document in top 5 0.57 (same as plain TF-IDF), faithfulness 1.00.
* 5 of 7 answerable questions were refused: everyday words the documents never use and the word
  list didn't contain (boss / OK / holiday, supplier / legit, double-checks, outage, lessons-learned).
  Safe, but not useful.
* 2 of 3 no-answer traps got a PARTIAL answer instead of a refusal -- the real problem:
  "leave days in the Madrid office" quoted other leave rules; "book a meeting room in Bengaluru"
  quoted the India tone profile.

Fixes made AFTER seeing v4 (so v4 no longer measures them fairly):
* One simple rule replaces the old share/weak-evidence rule: if the question uses a word the
  documents never use, KORA refuses and names the word (the "why" view says: add it to the company
  word list if it means something the documents say).
* Hyphenated words count as one ("part-time" must appear as such).
* A place is matched by its code when choosing sentences and checking coverage ("London" ->
  LOC-LON, which is how the leave policy says it); search itself is unchanged.
* Each sentence is scored with its section heading ("## Annual Leave").

After these, across all five question sets (51 questions): no question that should be refused got
an answer. Development 0.98 correct status; v1 0.80; v2 0.80; v3 0.70; v4 0.30 (all three traps
refused; all seven answerable questions refused -- their wording is not in the documents).

The honest limit of Phase 4: KORA understands the words the documents use, plus the company word
list. It never guesses, and it says which word it didn't know. Everyday wording is handled by
growing the word list from real refusals (logged in data/logs/answers.jsonl), or later by the
optional language-model step. A fair next test is questions written by someone other than the
builder.

## Held-out v5: 10 questions written by UT (independent author)

UT wrote 10 questions in his own words without looking at the documents (salary, promotion,
changing team, certifications, overtime, notice period, being late, reimbursement contact, talking
to the CEO, who to report to when the manager is away). Only 2 are documented (reimbursement; and,
loosely, communication with senior people); 8 should be refused.

Official result (system as shipped before this run): correct status 0.60 -- and THREE WRONG
ANSWERS, the most important finding of Phase 4 so far:
* "What is my salary structure?" -> ANSWERED from an org-chart line ("structure") plus an insurance
  line that happened to mention "salary level". Each word was found, but in different places.
* "What is the procedure to change the team?" -> ANSWERED with the Customer Onboarding gap. Every
  word of the question was on the "generic" list, so nothing was checked; and declared-gap
  paragraphs counted as "structured facts", which skip all checks.
* "I don't want to serve the notice period" -> PARTIAL, quoting an engineering rule ("1-week notice
  period to dependent squads"). Same words, different meaning (homonym).

Fixes (after seeing v5, so v5 is now spent):
* Weak evidence must hold together: when the best word match is weak, more than half of the main
  words must appear in ONE quoted line (with its heading), not scattered over lines or documents.
* If every word of a question is "generic", KORA checks the plain content words instead of nothing.
* Declared-gap paragraphs pass the same checks as any quote.
* "contact" is a question word ("who do I contact"), like "ask" or "tell".

After the fixes: v5 0.80, with the salary / change-team / late / overtime questions refused and the
reimbursement question answered. No regressions on development, v1-v4 (v1's H10, which had been
answered from the wrong documents, is now refused). Still open: the "notice period" homonym is
answered as PARTIAL -- word matching cannot tell the two meanings apart. This, and "stock", are the
known homonym limitation; the optional language-model step is the realistic fix.

## M6 and M7 (Phase 4 close)

* M6: `python -m src.ask` (one question or interactive; `--why`, `--as EMPLOYEE_ID`, `--top-k`) and the
  README's Phase 4 section.
* M7, access control: rules in the config (document pattern -> groups from the employees table).
  Enforced at four points: search, the router's plan, every answer line (a conflict with a hidden side
  is not shown at all), and a final pass over notes. Tested with 10 questions aimed at the hidden
  finance documents: for a user without access, no hidden document appears as a citation, retrieved
  chunk, routed candidate, line, note or log entry. A text in a document the user CAN read may still
  mention a hidden document's ID (e.g. "see PROC-FIN-003"); that is the readable document's own content.
  user=None means no filtering (backend and evaluation); the Phase 5 UI must always pass a user.
