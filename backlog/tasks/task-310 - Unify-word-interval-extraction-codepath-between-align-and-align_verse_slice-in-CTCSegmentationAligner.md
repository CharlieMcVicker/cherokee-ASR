---
id: TASK-310
title: >-
  Unify word interval extraction codepath between align and align_verse_slice in
  CTCSegmentationAligner
status: Done
assignee:
  - '@myself'
created_date: '2026-09-12 20:26'
updated_date: '2026-09-12 20:28'
labels: []
dependencies: []
ordinal: 326000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor CTCSegmentationAligner so that align_verse_slice and align share a single unified helper method for extracting word intervals, character state emissions, and word confidence scores from CTC trellis timings and state lists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extract a shared helper method _extract_word_intervals in CTCSegmentationAligner
- [x] #2 Refactor both align and align_verse_slice to use the unified helper method
- [x] #3 Ensure all existing unit tests in test_ctc_aligner.py pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define _extract_word_intervals helper method on CTCSegmentationAligner with full typing, handling word interval boundary extraction, emitted word recovery, and acoustic confidence calculation.\n2. Update align_verse_slice to use _extract_word_intervals.\n3. Update align to use _extract_word_intervals across chunks.\n4. Run pytest test suite to ensure all tests pass and no regressions are introduced.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored CTCSegmentationAligner in transcription/alignment/ctc_aligner.py by extracting _extract_word_intervals, unifying the word interval extraction, emitted token decoding, and acoustic confidence scoring logic between align and align_verse_slice. Verified all 246 unit tests in the suite pass.
<!-- SECTION:FINAL_SUMMARY:END -->
