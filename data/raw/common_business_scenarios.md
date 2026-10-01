**Document ID:** DOC-SCEN-001
**Document Title:** Common Cross-Department Business Scenarios
**Department:** Cross-Department
**Document Type:** Scenario Reference
**Owner/Role:** Operations Analyst
**Version:** 1.2
**Effective Date:** 2025-05-15
**Status:** Active
**Related Processes:** Multiple (referenced per scenario)
**Related Roles:** Multiple

---

# Common Cross-Department Business Scenarios

**Scenario: A customer wants to add NexCRM to their existing NexFlow contract.**
This is an expansion sale. The customer's Customer Success Manager loops in an Account
Executive; the AE runs standard proposal/quotation (PROC-SALES-002); Implementation
handles the NexCRM-specific onboarding (PROC-IMPL-001) while the CSM continues owning
the overall relationship.

**Scenario: A P1 incident affects one strategic customer for 30 hours.**
Standard incident management applies (PROC-SUP-002). Because duration exceeds 24 hours
for an active P1, the customer's CSM should escalate to VP Customer Success per
PROC-CS-001, Level 2 — this is a judgment call made by the CSM based on relationship
context, not an automatic trigger.

**Scenario: A new Implementation Consultant joins and needs access to customer
production tenants.**
Requires Implementation Manager approval (per PROC-IT-001's role-specific access table),
separate from their standard Day-1 access (email, chat, HR portal, wiki) which is
automatic.

**Scenario: An AE closes a deal but forgets to document the agreed go-live date in the
handoff record.**
This triggers the known gap KG-01 in PROC-IMPL-001 — there is no formal fallback
procedure; the Implementation Consultant currently has to chase the AE directly to fill
the gap before kickoff can be scheduled confidently.
