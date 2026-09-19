**Document ID:** DOC-TG-001
**Document Title:** Common Troubleshooting Guide
**Department:** Technical Support
**Document Type:** Troubleshooting Guide
**Owner/Role:** Support Team Lead
**Version:** 1.5
**Effective Date:** 2025-04-01
**Status:** Active
**Related Processes:** PROC-SUP-001 (Customer Support Procedures)
**Related Roles:** Support Engineer L1, L2, L3

---

# Common Troubleshooting Guide

**TG-01 — Customer reports NexFlow reorder alerts not firing**
Check: (1) reorder point is set on the item, (2) NexAuto automation for reorder alerts
is enabled for the tenant, (3) the item's warehouse has stock tracking enabled. Most
common root cause: reorder point was never set, mistaken for a product defect.

**TG-02 — Customer reports NexConnect sync failures**
Check the sync error log first (visible to L2+). Most common causes: source system
credentials expired, or a field mapping broke after the customer changed their source
system's schema without notifying Nexora. Escalate to Solutions Architect if the
connector is a custom build.

**TG-03 — Customer expects real-time NexView dashboard updates but sees hourly refresh**
This is expected behavior on the standard tier — real-time (sub-15-minute) refresh is a
"NexView Pro" tier feature. Confirm tier before treating as a bug. See DOC-PROD-NEXVIEW,
Known Limitation.

**TG-04 — Customer locked out of NexCRM after a password reset**
Standard reset flow issue; L1 can resolve directly via the admin panel. Escalate to L2
only if the customer's account shows SSO configuration (rare, mostly larger customers).

**TG-05 — New employee cannot access the analytics warehouse despite manager approval**
Confirm IT has completed provisioning — approval and provisioning are two separate
steps (see PROC-IT-001). A common false alarm is that approval was granted but IT
ticket wasn't filed.
