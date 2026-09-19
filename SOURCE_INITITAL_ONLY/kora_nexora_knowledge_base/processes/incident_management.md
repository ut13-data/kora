**Document ID:** PROC-SUP-002
**Document Title:** Incident Management Process
**Department:** Technical Support, IT / Security
**Document Type:** Policy / SOP
**Owner/Role:** IT Manager / Support Team Lead (joint ownership)
**Version:** 2.0
**Effective Date:** 2025-04-01
**Status:** Active
**Related Processes:** PROC-SUP-001 (Customer Support), PROC-IT-002 (Data Security Policy)
**Related Roles:** DevOps Engineer, IT Manager, Support Team Lead, CISO

---

# Incident Management Process

## Priority Definitions

| Priority | Definition | Response SLA (Standard) |
|---|---|---|
| P1 | Production down / data loss risk, affecting multiple customers or one customer's core operations | 4 business hours |
| P2 | Major feature unusable, workaround exists | 1 business day |
| P3 | Minor feature issue, limited impact | 3 business days |
| P4 | Cosmetic / low impact | Best effort |

## Steps

1. **Detection** — Via monitoring alert (DevOps), Support ticket escalation, or direct
   customer report.
2. **Incident declaration** — Any P1 is declared a formal incident; IT Manager or
   on-call DevOps Engineer opens an incident channel.
3. **Response** — Relevant Engineering squad and DevOps investigate; Support Team Lead
   owns customer communication cadence (updates at least every 2 hours during an active
   P1).
4. **Resolution** — Fix deployed or mitigation applied; incident downgraded or closed.
5. **Post-incident review** — Required for all P1 incidents within 5 business days;
   owned by the Engineering Manager of the affected squad, reviewed by CISO if the
   incident involved any data exposure risk.

## Escalation to Security
Any incident with suspected unauthorized data access must be escalated immediately to
the CISO, in parallel with standard incident response — see PROC-IT-002.
