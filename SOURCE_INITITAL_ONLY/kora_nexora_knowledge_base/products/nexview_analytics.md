**Document ID:** DOC-PROD-NEXVIEW
**Document Title:** NexView Analytics — Product Documentation
**Department:** Product
**Document Type:** Product Documentation
**Owner/Role:** Product Manager (NexView)
**Version:** 4.1
**Effective Date:** 2025-04-10
**Status:** Active
**Related Processes:** PROC-ENG-001 (SDLC), PROC-IMPL-001 (Customer Onboarding)
**Related Roles:** Product Manager, BI Developer, Implementation Consultant

---

# NexView Analytics

Nexora's flagship business analytics and reporting platform. Customers connect their
operational data (often via NexConnect) and build dashboards covering sales, inventory,
finance and operations KPIs without needing a dedicated BI team.

## Core Capabilities

- Pre-built report templates by industry vertical (pharma distribution, manufacturing,
  retail)
- Custom dashboard builder (drag-and-drop)
- Scheduled report exports (PDF, Excel, email delivery)
- Role-based data visibility within a tenant
- Embedded analytics API for customers who want NexView charts inside their own portals

## Typical Customer Profile

Mid-sized distribution or manufacturing businesses (50–2,000 employees) without an
in-house data team. Balaji Pharma Distribution is a representative reference customer
for the pharma distribution vertical template.

## Common Configuration Dependencies

NexView requires a data source connection, most commonly configured via NexConnect
during customer onboarding (PROC-IMPL-001). Reports cannot be built until at least one
verified data source is live.

## Known Limitation

Real-time (sub-15-minute) refresh is only available on the "NexView Pro" tier; standard
tier refreshes hourly. This is a frequent source of customer confusion during
onboarding — see troubleshooting_guide.md, TG-03.
