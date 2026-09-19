**Document ID:** PROC-IT-001
**Document Title:** Access Control Policy
**Department:** IT / Security
**Document Type:** Policy
**Owner/Role:** CISO
**Version:** 2.3
**Effective Date:** 2025-04-15
**Status:** Active
**Related Processes:** PROC-IT-002 (Data Security Policy), PROC-HR-001 (Employee Onboarding)
**Related Roles:** IT Support Specialist, Security Analyst, Data & Analytics Manager, CISO

---

# Access Control Policy

## Principle
Access to internal systems and customer data is granted on a least-privilege,
role-based basis and must be requested and approved — it is never granted by default
beyond standard company-wide tools (email, chat, HR system).

## Standard Access (granted at Day 1 for all employees)
Email, internal chat, HR self-service portal, company wiki.

## Role-Specific Access (requires request + approval)

| System | Approver |
|---|---|
| Analytics production data warehouse | Data & Analytics Manager + IT provisioning |
| Customer production tenants (Support tooling) | Support Team Lead |
| Customer production tenants (Implementation tooling) | Implementation Manager |
| Production code deployment | Engineering Manager |
| Billing/finance systems | Finance Manager |

## Analytics Environment Access
New Data Analysts require explicit Data & Analytics Manager approval before IT
provisions production data warehouse access — this is not automatic at hire, even for
the Data & Analytics department, because the warehouse contains customer data alongside
internal data.

## Access Review
All role-specific access is reviewed quarterly by IT/Security in coordination with each
approving manager; unused access (no login in 90 days) is automatically revoked pending
manager confirmation.
