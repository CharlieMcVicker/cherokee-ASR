---
id: TASK-185
title: Fix Pyright errors in transcription.utils module
status: Done
assignee:
  - '@pyright-fixer'
created_date: '2026-07-27 17:53'
updated_date: '2026-07-27 18:12'
labels: []
dependencies: []
ordinal: 181000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix 5 pyright type errors in  (reportArgumentType, reportCallIssue).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All pyright errors in transcription/utils are resolved
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dead imports (shutil in evaluation.py, respell_consonants in syllabary_map.py), committed dead code separately, and resolved all 5 Pyright type errors across transcription/utils. Verified with 0 Pyright errors and all 46 unit tests passing.
<!-- SECTION:FINAL_SUMMARY:END -->
