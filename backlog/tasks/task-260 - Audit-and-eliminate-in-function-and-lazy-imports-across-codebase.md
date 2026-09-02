---
id: TASK-260
title: Audit and eliminate in-function and lazy imports across codebase
status: To Do
assignee: []
created_date: '2026-09-02 15:01'
labels:
  - code-smell
  - refactor
  - code-quality
dependencies: []
ordinal: 262000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code smell identified in PR #3 review: imports defined inside functions instead of at module top level. Audit all Python files in the repository (e.g. inference, training, new_testament, syllabary_enrichment, tests) to move imports to top-level and resolve any circular dependencies cleanly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identify all in-function / lazy imports in transcription and tests
- [ ] #2 Move imports to top-level module scope
- [ ] #3 Ensure no circular import regressions or startup delays
- [ ] #4 Verify all tests pass and pyright typechecking succeeds
<!-- AC:END -->
