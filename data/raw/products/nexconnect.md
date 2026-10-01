**Document ID:** DOC-PROD-NEXCONNECT
**Document Title:** NexConnect — Product Documentation
**Department:** Product
**Document Type:** Product Documentation
**Owner/Role:** Product Manager (NexConnect)
**Version:** 2.4
**Effective Date:** 2025-03-15
**Status:** Active
**Related Processes:** PROC-ENG-001 (SDLC), PROC-IMPL-001 (Customer Onboarding)
**Related Roles:** Product Manager, Solutions Architect, Data Engineer

---

# NexConnect

Data integration / middleware layer that connects a customer's existing systems (ERP,
accounting software, e-commerce platforms) to Nexora products, most commonly NexView
Analytics and NexFlow.

## Core Capabilities

- Pre-built connectors for common ERP and accounting systems
- Custom connector framework (REST/SFTP-based) for bespoke integrations
- Scheduled and real-time (webhook-based) sync options
- Data mapping and transformation layer
- Sync error monitoring and alerting

## Typical Customer Profile

Any customer connecting an external data source to NexView Analytics or NexFlow.
NexConnect is rarely sold standalone; it is foundational infrastructure for most
multi-product deployments.

## Ownership Note

NexConnect implementation work requires a Solutions Architect, not a standard
Implementation Consultant, for any custom connector build (as opposed to a pre-built
connector, which any Implementation Consultant can configure).
