**Document ID:** PROC-ENG-001
**Document Title:** Software Development Lifecycle (SDLC)
**Department:** Software Engineering
**Document Type:** SOP
**Owner/Role:** VP Engineering
**Version:** 3.0
**Effective Date:** 2025-01-08
**Status:** Active
**Related Processes:** PROC-ENG-002 (Bug Reporting & Escalation)
**Related Roles:** Product Manager, Software Engineer, Engineering Manager, QA Engineer, DevOps Engineer

---

# Software Development Lifecycle

## Purpose
Standardize how features move from specification to production release across all
Engineering squads.

## Steps

1. **Spec handoff** — Product Manager delivers a written spec to the squad's Engineering
   Manager.
2. **Sprint planning** — Squad breaks the spec into tickets, estimates, and commits to a
   two-week sprint.
3. **Development** — Engineers implement against a feature branch; Senior Engineers
   perform code review before merge.
4. **QA verification** — QA Engineer runs the test plan against a staging environment.
5. **Release** — DevOps Engineer deploys to production via the CI/CD pipeline, typically
   on a Wednesday release window.
6. **Post-release monitoring** — Squad monitors error rates for 48 hours post-release.

## Definitions
"Defect" (Engineering's term) = confirmed deviation from expected behavior, equivalent to
Support's "bug" — see glossary.md TN-008.
