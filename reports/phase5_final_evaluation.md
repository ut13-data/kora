# Phase 5 compared with Phase 4

Made by `python -m src.evaluation.baseline --report reports/phase5_final_evaluation.md`.

**No question got worse.** FIXED: 12. CHANGED with the same correctness: 13.

How to read it:

* *before* is the Phase 4 result frozen at the start of Phase 5 (`data/evaluation/baseline_phase4.json`).
* *names* checks that the people, roles or teams the question is about appear in the answer.
* *no leak* checks that no hidden document is quoted, cited or named for that employee. A new leak always counts as BROKEN.
* *cite prec* leaves out company-record citations (CSV rows), which have no document chunk.
* `cited [...] -> []` means the answer now cites a document header or a company record instead of a passage.
* The graph sets were written during Phase 5 by the builder; held-out H02, H04 and H09 were seen during building.

```
Before -> now (+ better, - worse)

Set                   status       key fact          names        no leak      cite prec       faithful          hit@5
development       0.98->0.98     1.00->1.00       n/a->n/a     1.00->1.00     0.93->0.93     1.00->1.00     1.00->1.00
graph_dev       0.59->1.00 +   0.33->1.00 +   0.12->1.00 +     1.00->1.00   0.95->1.00 +     1.00->1.00     0.93->0.93
graph_heldout   0.75->1.00 +     1.00->1.00   0.19->1.00 +     1.00->1.00   0.77->1.00 +     1.00->1.00     0.88->0.88
heldout_v1        0.70->0.70     0.62->0.62       n/a->n/a     1.00->1.00     0.93->0.93     1.00->1.00     0.89->0.89
heldout_v2        0.80->0.80     0.88->0.88       n/a->n/a     1.00->1.00     0.94->0.94     1.00->1.00     1.00->1.00
heldout_v3        0.70->0.70     0.62->0.62       n/a->n/a     1.00->1.00     0.92->0.92     1.00->1.00     1.00->1.00
heldout_v4        0.30->0.30     0.00->0.00       n/a->n/a     1.00->1.00       n/a->n/a     1.00->1.00     0.57->0.57
heldout_v5        0.80->0.80     0.00->0.00       n/a->n/a     1.00->1.00     0.50->0.50     1.00->1.00     0.50->0.50

BROKEN (right before, wrong now): 0

FIXED: 12
  graph_dev/G01 not_found -> answered  cited [] -> ['PROC-IMPL-001']  | Which teams are involved in customer onboarding?
  graph_dev/G03 not_found -> answered  cited [] -> ['PROC-IMPL-001']  | Which process could be affected if the NexConnect integration procedure changes?
  graph_dev/G04 not_found -> partial  cited [] -> ['PROC-IMPL-001', 'PROC-SALES-002']  | Why is the customer onboarding process delayed?
  graph_dev/G07 not_found -> answered  cited [] -> ['PROC-SALES-002']  | Which process starts when a deal is closed?
  graph_dev/G11 not_found -> answered  | Which department does the Implementation Manager belong to?
  graph_dev/G12 not_found -> answered  | What does Lena Hoffmann do?
  graph_dev/G15 not_found -> answered  cited [] -> ['PROC-IMPL-001']  | Who gave final approval for customer onboarding before 2025?
  graph_dev/G16 not_found -> answered  cited [] -> ['PROC-SUP-002']  | Which roles take part in handling a P1 incident?
  graph_dev/G18 not_found -> answered  cited [] -> ['PROC-FIN-003']  | Who takes part in the invoice and payment process?
  graph_heldout/H01 not_found -> answered  cited [] -> ['PROC-HR-001']  | Who is involved in the employee onboarding process?
  graph_heldout/H02 not_found -> partial  cited [] -> ['PROC-HR-001', 'PROC-IT-001', 'PROC-IT-002', 'PROC-SUP-002']  | If the Data Security Policy changes, which processes might be affected?
  graph_heldout/H08 partial -> answered  cited ['DOC-DEPT-FIN', 'DOC-DEPT-HR', 'PROC-FIN-002'] -> []  | Who heads the department that owns NexFlow?

CHANGED (same correctness): 13
  graph_dev/G02 answered -> answered  cited ['PROC-FIN-002'] -> []  | Who owns the procurement process?
  graph_dev/G06 answered -> answered  cited ['DOC-DEPT-SUP', 'PROC-ENG-002', 'PROC-SUP-001'] -> ['PROC-SUP-001']  | What happens after a support engineer confirms a product defect?
  graph_dev/G08 partial -> answered  | Which policies govern the incident management process?
  graph_dev/G09 answered -> answered  cited ['DOC-DEPT-PROD'] -> []  | Which products does the Product department own?
  graph_dev/G10 answered -> answered  cited ['DOC-DEPT-FIN'] -> []  | Who is the head of the Finance department?
  graph_dev/G13 answered -> answered  cited ['PROC-IT-002', 'PROC-SUP-002'] -> []  | Which processes does the CISO own?
  graph_dev/G14 answered -> answered  cited ['DOC-PROD-NEXCRM'] -> ['PROC-SALES-001', 'PROC-SALES-002']  | Which processes use NexCRM?
  graph_dev/G22 answered -> answered  cited ['DOC-DEPT-SUP'] -> []  | Which department owns the Bug Reporting and Escalation process?
  graph_heldout/H03 answered -> answered  cited ['PROC-CS-001', 'PROC-SUP-002'] -> []  | Who owns the incident management process?
  graph_heldout/H05 answered -> answered  cited ['DOC-DEPT-SALES', 'DOC-PROD-CUSTDEV', 'PROC-SALES-001'] -> ['PROC-IMPL-001', 'PROC-SALES-001', 'PROC-SALES-002']  | Which processes is the Account Executive part of?
  graph_heldout/H06 answered -> answered  cited ['DOC-DEPT-SALES'] -> []  | Which processes does the Sales department own?
  graph_heldout/H07 answered -> partial  | What comes after lead qualification?
  graph_heldout/H09 answered -> answered  cited ['DOC-DEPT-CS'] -> []  | Which department is responsible for customer escalations?
```
