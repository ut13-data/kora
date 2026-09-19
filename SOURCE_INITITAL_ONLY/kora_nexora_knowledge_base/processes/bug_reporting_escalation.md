**Document ID:** PROC-ENG-002
**Document Title:** Bug Reporting and Escalation Process
**Department:** Software Engineering, Technical Support
**Document Type:** SOP
**Owner/Role:** Support Team Lead / Engineering Manager (joint ownership)
**Version:** 2.2
**Effective Date:** 2025-03-20
**Status:** Active
**Related Processes:** PROC-SUP-001 (Customer Support Procedures), PROC-ENG-001 (SDLC)
**Related Roles:** Support Engineer L3, Engineering Manager, Software Engineer

---

# Bug Reporting and Escalation Process

## Purpose
Define how a customer-reported issue confirmed as a product defect moves from Support
into Engineering's backlog.

## Steps

1. **Confirmation** — A Support Engineer L3 reproduces the issue and confirms it is a
   genuine product defect (not user error or configuration issue).
2. **Defect logging** — L3 logs the defect in the Engineering defect tracker with
   severity, reproduction steps, and affected customer(s).
3. **Severity triage** — Engineering Manager for the relevant squad triages severity
   (P1–P4, aligned with incident priority definitions in PROC-SUP-002) within 1 business
   day.
4. **Fix scheduling** — P1/P2 defects are pulled into the current or next sprint. P3/P4
   are backlog-prioritized in the normal roadmap process.
5. **Customer communication** — Support (not Engineering) owns all customer
   communication about defect status throughout.
6. **Resolution & verification** — Once fixed and released, the original L3 Support
   Engineer verifies the fix against the customer's reported scenario before closing the
   ticket.

## Known Gap — KG-02
This process does not document what happens when a customer reports a bug on a
**deprecated product version** that is no longer receiving active development (e.g., a
legacy NexFlow configuration mode retired from new sales but still running for a small
number of legacy customers). In practice, these reports are currently triaged
case-by-case by the Engineering Manager with no documented standard, which has led to
inconsistent customer communication in at least two known cases.
