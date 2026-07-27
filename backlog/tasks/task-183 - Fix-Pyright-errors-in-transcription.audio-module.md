---
id: TASK-183
title: Fix Pyright errors in transcription.audio module
status: Done
assignee:
  - '@pyright-fixer'
created_date: '2026-07-27 17:53'
updated_date: '2026-07-27 18:07'
labels: []
dependencies: []
ordinal: 179000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix 15 pyright type errors in  (reportMissingImports, reportUnboundVariable, unknown, etc.).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All pyright errors in transcription/audio are resolved
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dead code files transcription/audio/extract_from_elan.py and transcription/audio/segmentation.py in a separate commit, added full type annotations and cast helpers in extract.py and segment.py, and verified 0 Pyright errors and 46/46 passing pytest tests.
<!-- SECTION:FINAL_SUMMARY:END -->
