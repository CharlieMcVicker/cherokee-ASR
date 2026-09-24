---
id: TASK-330
title: Implement strongly-typed Orthography Enums and idempotent conversion maps
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 19:56'
updated_date: '2026-09-14 20:00'
labels: []
dependencies: []
ordinal: 346000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define explicit domain enums for Cherokee orthographic representations (Cherokee Syllabary, Latin Citation Toned, Latin Citation Toneless, ASR Native d/g, ASR Respelled t/th) and implement idempotent pure transformation maps to eliminate double-normalization errors across the alignment and inference pipelines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define Orthography enum representing all surface and acoustic representations in the domain
- [x] #2 Implement pure, idempotent transformation maps between orthography types
- [x] #3 Refactor normalize_phonetics_for_alignment and alignment normalizers to use typed orthography dispatchers with idempotency guarantees
- [x] #4 Add comprehensive unit tests verifying idempotency and cross-orthography conversion correctness
- [x] #5 Ensure pytest suite passes with zero regressions
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Orthography enum and pure convert_orthography map in transcription.utils.orthography with strict short-circuiting when source == target. Refactored normalizers.py to use typed orthography conversion, eliminating double-normalization and guaranteeing idempotency. Added unit tests in test_orthography.py; all 271 pytest tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
