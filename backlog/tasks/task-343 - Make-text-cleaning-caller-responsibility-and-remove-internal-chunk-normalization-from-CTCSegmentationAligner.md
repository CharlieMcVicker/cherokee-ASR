---
id: TASK-343
title: >-
  Make text cleaning caller responsibility and remove internal chunk
  normalization from CTCSegmentationAligner
status: Done
assignee:
  - '@myself'
created_date: '2026-09-16 17:44'
updated_date: '2026-09-16 17:47'
labels: []
dependencies: []
ordinal: 359000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove chunk_normalizer from CTCSegmentationAligner so the aligner acts as a pure alignment engine on pre-normalized target phonetics. Update all callers (new_testament pipeline, ingestion, alignment CLI, scripts, tests) to clean and normalize text at the outer I/O boundary, eliminating double-normalization bugs (e.g. DG -> TTH applied multiple times).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove chunk_normalizer from CTCSegmentationAligner and ensure align() only splits input chunk text directly without re-converting orthography
- [x] #2 Update all callers across pipeline, CLI, scripts, and tests to normalize text at the caller boundary
- [x] #3 Verify Mark Chapter 1 realignment preserves plain stops (e.g. tsinikvnv) without spurious double-conversion aspiration
- [x] #4 Run full test suite (pytest) and pyright to ensure zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove chunk_normalizer from CTCSegmentationAligner in transcription/alignment/ctc_aligner.py and make align() split input chunk text directly.
2. Find all call sites across transcription/ and scripts/ and update them to ensure caller handles all text cleaning/normalization.
3. Realign Mark Chapter 1 and verify that plain stops (e.g. tsinikvnv, nahskiya, anatole...) are preserved cleanly without double-conversion corruption.
4. Run pytest (all 275 tests) and pyright transcription.
5. Finalize task and update acceptance criteria.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed internal chunk_normalizer from CTCSegmentationAligner, delegating all text cleaning and orthography conversion to the caller boundary. Updated all pipeline callers and test harnesses. Verified Mark Chapter 1 realignment eliminates double-normalization aspiration bugs (preserving plain stops such as tsinikvnv). Full pytest suite (275/275) and pyright (0 errors) pass.
<!-- SECTION:FINAL_SUMMARY:END -->
