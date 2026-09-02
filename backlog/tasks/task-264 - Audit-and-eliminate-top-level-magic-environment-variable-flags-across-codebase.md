---
id: TASK-264
title: Audit and eliminate top-level magic environment variable flags across codebase
status: Done
assignee:
  - '@agent-env-audit'
created_date: '2026-09-02 15:33'
updated_date: '2026-09-02 15:39'
labels:
  - code-smell
  - refactor
  - code-quality
dependencies: []
priority: low
ordinal: 266000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit the codebase for module-level os.environ mutations (such as KMP_DUPLICATE_LIB_OK and PYTORCH_ENABLE_MPS_FALLBACK). Ensure environment settings are properly configured via runtime flags or CLI runners rather than top-level side effects in import headers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify all module-level os.environ mutations across the codebase
- [x] #2 Remove unnecessary magic flags from library modules
- [x] #3 Ensure runtime CLI entrypoints handle necessary environment configurations cleanly
- [x] #4 Verify all unit tests pass and pyright reports 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audited and removed top-level magic environment variables across codebase
<!-- SECTION:FINAL_SUMMARY:END -->
