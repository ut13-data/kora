**Document ID:** DOC-CORP-003
**Document Title:** Nexora Systems — Internal Glossary & Terminology Notes
**Department:** Executive / Corporate
**Document Type:** Glossary
**Owner/Role:** VP Product
**Version:** 1.6
**Effective Date:** 2025-02-10
**Status:** Active
**Related Processes:** All
**Related Roles:** All Employees

---

# Glossary

| Term | Definition |
|---|---|
| ARR | Annual Recurring Revenue |
| Go-Live | The date a customer's Nexora product configuration becomes the system of record in production |
| Onboarding | Overloaded term — see Terminology Note TN-002 |
| SOW | Statement of Work, attached to a signed proposal |
| Tenant | A single customer's isolated instance/configuration within a Nexora product |
| Ticket | A logged customer-reported issue in the support system |
| P1 / P2 / P3 / P4 | Support/Incident priority levels, see PROC-SUP-002 |
| SDR | Sales Development Representative |
| AE | Account Executive |
| CSM | Customer Success Manager |
| BODS-style migration | Internal shorthand for large batch data-migration work, referencing the SAP BusinessObjects Data Services heritage of several senior Data & Analytics hires |

## Terminology Notes (Known Inconsistencies)

**TN-002 — "Onboarding" is used for two different things.**
HR uses "onboarding" to mean *employee* onboarding (PROC-HR-001). Implementation and
Customer Success use "onboarding" to mean *customer* onboarding (PROC-IMPL-001). Always
disambiguate: "employee onboarding" vs. "customer onboarding." Several older Support
tickets and Slack references use bare "onboarding" and must be interpreted from context.

**TN-004 — "Client" vs. "Customer" vs. "Account" vs. "Customer Account."**
- Sales and Marketing materials predominantly say **"client."**
- Customer Success and Support predominantly say **"customer."**
- Finance and the billing system use **"account"** or **"customer account"** as the
  formal record-keeping unit (one account can technically map to multiple
  tenants/contracts).
These are not always perfectly interchangeable — a "client" in a Sales context may refer
to a prospective company before a contract exists, whereas "account" in Finance always
refers to a billable, contracted entity. There is no single company-wide style guide
enforcing one term; this is a long-standing, low-priority inconsistency.

**TN-006 — "Project Lead" vs. "Implementation Manager."**
See organization.md, "Note on Titles." Pre-2025-03-01 documents say "Project Lead";
current documents say "Implementation Manager." Same role.

**TN-008 — "Bug" vs. "Defect" vs. "Issue."**
Engineering's SDLC documentation (PROC-ENG-002) uses "defect." Support-facing
documentation (PROC-SUP-001) uses "bug." The ticketing system field is literally labeled
"Issue Type." All three refer to the same underlying concept: a confirmed deviation from
expected product behavior.
