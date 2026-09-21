---
id: TASK-354
title: Remove legacy shims and dict access in normalizers and ASRResult
status: Done
assignee: []
created_date: '2026-09-21 17:04'
updated_date: '2026-09-21 17:10'
labels: []
dependencies: []
ordinal: 380000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code review identified Clean Break violations: deprecated alias normalize_text_for_alignment and dict subscripting shims in ASRResult.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove normalize_text_for_alignment from transcription/alignment/normalizers.py
- [x] #2 Remove unused imports from transcription/alignment/ingestion.py
- [x] #3 Remove __getitem__ and get methods from ASRResult in transcription/models/asr_model.py and verify direct attribute access
- [x] #4 All existing tests pass with 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed legacy compatibility shims (normalize_text_for_alignment) and dict subscripting from ASRResult. Verified all tests and pyright pass.
<!-- SECTION:FINAL_SUMMARY:END -->
