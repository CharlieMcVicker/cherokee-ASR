---
id: TASK-180
title: Fix Pyright errors in transcription.inference module
status: Done
assignee:
  - '@pyright-fixer'
created_date: '2026-07-27 17:52'
updated_date: '2026-07-27 18:01'
labels: []
dependencies: []
ordinal: 176000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix 85 pyright type errors in  (reportCallIssue, unknown, reportArgumentType, reportPossiblyUnboundVariable, reportMissingImports, etc.).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All pyright errors in transcription/inference are resolved
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed unreferenced dead code files run_julie.py and long_recording.py (commit a3b9758 and c3e5368). Audited external interfaces and fixed all 85 Pyright type errors in transcription/inference (infer.py, batch.py, single.py, run.py, labeler.py). Verified with 0 Pyright errors and passing unit test suite.
<!-- SECTION:FINAL_SUMMARY:END -->
