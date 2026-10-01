**Document ID:** DOC-PROD-NEXAUTO
**Document Title:** NexAuto — Product Documentation
**Department:** Product
**Document Type:** Product Documentation
**Owner/Role:** Product Manager (NexAuto)
**Version:** 1.9
**Effective Date:** 2024-12-15
**Status:** Active
**Related Processes:** PROC-ENG-001 (SDLC)
**Related Roles:** Product Manager, Implementation Consultant

---

# NexAuto

Workflow automation engine used to build "if this, then that" style automations across
Nexora products and select third-party tools (email, Slack, common ERP webhooks).

## Core Capabilities

- Trigger-action automation builder (no-code)
- Cross-product triggers (e.g., NexFlow stock-low event → NexAuto → generate PO draft)
- Third-party webhook support
- Automation run history and error logging

## Typical Customer Profile

Existing NexFlow or NexCRM customers looking to reduce manual, repetitive process steps.
Rarely sold as a standalone first product.

## Known Limitation

NexAuto automations do not currently support conditional branching beyond two levels of
nesting. This is a common Product roadmap request; see product feedback backlog
(not included in this knowledge base export).
