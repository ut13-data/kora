# KORA Phase 1 — Nexora Systems Synthetic Knowledge Base

This is the complete synthetic knowledge universe for **Nexora Systems**, a fictional
mid-sized B2B software company (~137 employees, 4 offices: Munich, Berlin, Bengaluru,
Austin, London), built as the ingestion source for a future KORA (Knowledge Object
Retrieval Agent) system. No code, UI, RAG pipeline, or vector database has been built —
this is data only.

---

## 1. Complete File / Folder Structure

```
kora_knowledge_base/
├── README.md                              (this file)
├── company_profile.md                     DOC-CORP-001
├── organization.md                        DOC-CORP-002
├── glossary.md                            DOC-CORP-003
├── faq.md                                 DOC-FAQ-001
├── troubleshooting_guide.md               DOC-TG-001
├── common_business_scenarios.md           DOC-SCEN-001
├── new_joiner_care.md                     DOC-CARE-001   (Wow Factor 1)
├── germany_settling_in_checklist.md       DOC-CARE-002   (Wow Factor 1)
├── first_week_germany.md                  DOC-CARE-003   (Wow Factor 1)
├── first_90_days_germany.md               DOC-CARE-004   (Wow Factor 1)
├── communication_culture.md               DOC-COMM-001   (Wow Factor 2)
├── germany_workplace_tone.md              DOC-COMM-002   (Wow Factor 2)
├── india_workplace_tone.md                DOC-COMM-003   (Wow Factor 2)
├── us_workplace_tone.md                   DOC-COMM-004   (Wow Factor 2)
├── uk_workplace_tone.md                   DOC-COMM-005   (Wow Factor 2)
├── international_communication.md         DOC-COMM-006   (Wow Factor 2)
├── departments/
│   ├── sales.md               DOC-DEPT-SALES
│   ├── marketing.md           DOC-DEPT-MKT
│   ├── product.md             DOC-DEPT-PROD
│   ├── engineering.md         DOC-DEPT-ENG
│   ├── data_analytics.md      DOC-DEPT-DATA
│   ├── implementation.md      DOC-DEPT-IMPL
│   ├── customer_success.md    DOC-DEPT-CS
│   ├── technical_support.md   DOC-DEPT-SUP
│   ├── finance.md             DOC-DEPT-FIN
│   ├── hr.md                  DOC-DEPT-HR
│   ├── it_security.md         DOC-DEPT-IT
│   └── operations.md          DOC-DEPT-OPS
├── products/
│   ├── nexview_analytics.md               DOC-PROD-NEXVIEW
│   ├── nexflow.md                         DOC-PROD-NEXFLOW
│   ├── nexcrm.md                          DOC-PROD-NEXCRM
│   ├── nexauto.md                         DOC-PROD-NEXAUTO
│   ├── nexconnect.md                      DOC-PROD-NEXCONNECT
│   ├── custom_development.md              DOC-PROD-CUSTDEV
│   └── implementation_support_services.md DOC-PROD-IMPLSUP
├── processes/
│   ├── lead_qualification.md                              PROC-SALES-001
│   ├── proposal_quotation.md                               PROC-SALES-002
│   ├── customer_onboarding.md                              PROC-IMPL-001 (v2.0, current)
│   ├── customer_onboarding_v1_ARCHIVED.md                  PROC-IMPL-001 (v1.0, archived)
│   ├── sdlc.md                                             PROC-ENG-001
│   ├── bug_reporting_escalation.md                         PROC-ENG-002
│   ├── customer_support_procedure.md                       PROC-SUP-001
│   ├── incident_management.md                              PROC-SUP-002
│   ├── access_control_policy.md                            PROC-IT-001
│   ├── data_security_policy.md                             PROC-IT-002
│   ├── employee_onboarding.md                              PROC-HR-001 (v3.0, current)
│   ├── archived_new_hire_security_training_v1_ARCHIVED.md  PROC-HR-001-ADDENDUM (archived)
│   ├── leave_policy.md                                     PROC-HR-002
│   ├── expense_policy.md                                   PROC-FIN-001
│   ├── procurement_process.md                              PROC-FIN-002
│   ├── invoice_payment_process.md                          PROC-FIN-003
│   ├── customer_escalation_policy.md                       PROC-CS-001
│   └── internal_communication_guidelines.md                PROC-HR-003
└── data/
    ├── employees.csv       (137 rows)
    ├── departments.csv     (12 rows)
    ├── roles.csv           (48 distinct roles)
    ├── products.csv        (7 rows)
    ├── processes.csv       (19 rows, incl. archived versions)
    ├── documents.csv       (53 rows — master index of every .md document)
    └── locations.csv       (5 rows)
```

**Total: 50 Markdown documents + 7 CSV files.**

---

## 2. List of All Generated Documents

See `data/documents.csv` for the full machine-readable index (Document ID, Title,
Department, Type, Version, Status, Owner, file path) — every document above is a row
there, so this file is the canonical catalog a future ingestion script should start
from.

---

## 3. Entity Types Present

`Employee`, `Department`, `Role`, `Location`, `Product`, `Customer` (referenced, not
separately tabulated — e.g. Balaji Pharma Distribution, Rheinland Industrial Supply),
`Process`, `Policy`, `Document`, `Document Version`, `Checklist`, `Topic`,
`Terminology`, `Communication Style / Tone Profile`, `Known Gap`, `Known Conflict`.

---

## 4. Important Relationships

- **Employee → Role → Department** — `employees.csv.role_id/dept_id` → `roles.csv` / `departments.csv`
- **Employee → Manager** — `employees.csv.manager_id` (self-referential)
- **Employee → Location** — `employees.csv.location_id` → `locations.csv`
- **Location → New Joiner Checklist** — Germany locations (LOC-MUC, LOC-BER) →
  `DOC-CARE-002/003/004`; `employees.csv.international_relocation_germany` flags who this
  applies to
- **Employee → Communication Context** — `location_id` → matching `DOC-COMM-00x` tone
  profile
- **Role → Process** — e.g. Implementation Manager → PROC-IMPL-001 (approval authority)
- **Process → Product** — e.g. PROC-IMPL-001 ↔ NexFlow/NexView/NexConnect configuration
  dependencies
- **Process → Policy** — e.g. PROC-HR-001 (Employee Onboarding) ↔ PROC-IT-002 (Data
  Security, training requirement) ↔ PROC-IT-001 (Access Control)
- **Document → Document Version** — PROC-IMPL-001 v1.0 (archived) → v2.0 (active);
  PROC-HR-001-ADDENDUM v1.0 (archived) superseded by PROC-HR-001 v3.0
- **Department → Product ownership** — `products.csv.owning_dept_id`

---

## 5. Intentional Knowledge Conflicts

1. **Security training deadline (version conflict).** Archived
   `PROC-HR-001-ADDENDUM` v1.0 (2022-08-01) says "within 30 days." Current `PROC-HR-001`
   v3.0 (2025-06-01) says "within the first 10 working days." The current document is
   authoritative; the archived one is clearly marked Status: Archived.
2. **"Project Lead" vs. "Implementation Manager" (title/responsibility ambiguity).**
   `PROC-IMPL-001` v1.0 (archived, 2023) uses "Project Lead"; v2.0 (active, 2025-03-01)
   uses "Implementation Manager." Same role. Documented in `organization.md` and
   `glossary.md` TN-006.
3. **Terminology conflict: "Client" vs. "Customer" vs. "Account."** Sales/Marketing say
   "client"; Customer Success/Support say "customer"; Finance's billing system says
   "account"/"customer account." Documented in `glossary.md` TN-004.
4. **"Bug" vs. "Defect" vs. "Issue."** Support-facing docs say "bug," Engineering's SDLC
   docs say "defect," the ticketing system field says "Issue Type." Same concept.
   `glossary.md` TN-008.
5. **"Onboarding" is overloaded.** HR's employee onboarding (`PROC-HR-001`) vs.
   Implementation's customer onboarding (`PROC-IMPL-001`) both use the bare word
   "onboarding." `glossary.md` TN-002.

---

## 6. Knowledge Gaps (Deliberately Missing / Ambiguous)

- **KG-01** (`PROC-IMPL-001`): No documented fallback if the AE's "Deal Closed" handoff
  record is missing required fields (e.g., no agreed go-live date).
- **KG-02** (`PROC-ENG-002`): No documented process for a customer-reported bug on a
  **deprecated** product configuration no longer under active development.
- **Proposal process gap** (`PROC-SALES-002`): No formal escalation path for
  mid-negotiation go-live date changes after a proposal is already approved at a given
  discount tier.
- **Germany checklist gaps** (`germany_settling_in_checklist.md`): Address registration,
  health insurance, and tax registration sections all explicitly say "Check the current
  official guidance for your location and circumstances" rather than fabricating
  government procedure detail.
- **Leave policy gap** (`PROC-HR-002`): Parental leave section explicitly does not
  enumerate country-specific entitlements; defers to HR Generalist / official guidance.
- **Customer escalation judgment gap** (`PROC-CS-001`): A P1 incident does not
  *automatically* trigger a Level 2 executive escalation — it is explicitly left to CSM
  judgment, with no hard rule.

---

## 7. KORA Test Scenarios (Future Questions a New Employee Could Ask)

**Company / Role / Process knowledge**
1. "I'm a new Data Analyst. Who approves access to the analytics environment?"
2. "Who does final go-live sign-off for a customer implementation, and has that role's
   name changed recently?"
3. "What's the difference between how Sales and Finance refer to our customers?"
4. "What happens after a deal closes — who do I hand it off to and what do they need
   from me?"
5. "Who owns the Customer Escalation Policy, and when does a technical incident become a
   customer escalation?"

**Product knowledge**
6. "A customer says their NexView dashboard isn't updating in real time — is that a
   bug?"
7. "Does NexFlow support batch/lot tracking, and can it be turned on after go-live?"

**HR / onboarding**
8. "How long do I have to complete security training as a new hire?"
9. "Two documents give me different deadlines for security training — which one is
   current?"
10. "I'm joining Nexora in Munich. What should I take care of during my first month
    outside of work?"
11. "I haven't done my address registration yet — what does the Nexora checklist say I
    should do?"
12. "What are the three things people commonly forget during their first month in
    Germany?"

**Cultural communication**
13. "How should I phrase a deadline-driven request to my German manager?"
14. "My colleague in Bengaluru said 'when you get a chance' about something I think is
    urgent — how should I read that?"
15. "Does Nexora expect everyone from a given country to communicate the same way?"

**Cross-department / conflicting or missing information**
16. "Who is the 'Project Lead' mentioned in this old onboarding doc — is that still a
    real role?"
17. "A customer reported a bug on a product version we don't actively develop anymore —
    what's the process?"
18. "What happens if the sales handoff record is missing the agreed go-live date?"

---

## 8. Why This Dataset Tests KORA

- **Entity recognition & relationship discovery**: `employees.csv` alone requires
  resolving Employee → Role → Department → Location → Manager chains across 137 rows
  and 48 distinct roles, with no single document containing the full picture.
- **Document versioning & conflict detection**: Three deliberate, clearly-dated
  version pairs (security training deadline, Project Lead/Implementation Manager title,
  PROC-IMPL-001 v1 vs v2) require a system to determine *which document is current* —
  not just find matching text.
- **Terminology normalization**: Four documented terminology inconsistencies
  (client/customer/account, bug/defect/issue, onboarding, Project Lead/Implementation
  Manager) require entity resolution across departments that don't share vocabulary.
- **Knowledge-gap detection**: Six explicitly flagged gaps test whether a system
  correctly reports "this isn't documented" rather than fabricating an answer —
  including the Germany checklist's repeated, deliberate "check official guidance"
  deferrals.
- **Contextual/employee personalization**: The `international_relocation_germany` flag
  and location-keyed checklist/tone-profile documents test whether a system can route
  the same underlying question ("what should I do this week?") differently depending on
  who's asking.
- **Cultural tone adaptation without fact distortion**: The four-layer FACT / COMPANY
  PREFERENCE / CULTURAL GUIDANCE / INDIVIDUAL PREFERENCE framework in
  `communication_culture.md` is a genuinely hard reasoning test — a naive system might
  let "cultural guidance" leak into and distort a factual answer, which this dataset is
  specifically built to catch.
- **Realism**: IDs are consistent across every CSV and every document's metadata header;
  nothing is a disconnected, one-off fictional fact — every process, product, role and
  policy is cross-referenced from at least one other document, the way a real internal
  wiki accretes over years rather than being written in one sitting.
