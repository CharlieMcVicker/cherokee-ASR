---
id: TASK-152
title: Map GT text onto exact ASR emission token boundaries
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:38'
updated_date: '2026-07-23 14:39'
labels: []
dependencies: []
ordinal: 148000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor aligner word interval assignment to preserve exact ASR emission token start_time and end_time boundaries, mapping ground-truth text (single or fused GT words) directly onto the raw ASR token intervals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Word intervals preserve exact raw ASR emission token timestamps
- [x] #2 GT text cleanly mapped onto ASR emission token boundaries
- [x] #3 Tests pass for ASR boundary preserving alignment
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refactored _align_words_char_range in aligner.py to map GT text directly onto raw ASR emission token intervals while preserving exact start_time and end_time boundaries.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Mapped GT words onto exact ASR emission token boundaries in transcription.timestamping. Refactored _align_words_char_range to preserve exact raw emission token start_time and end_time timestamps, mapping GT text (or fused GT words) directly onto the ASR emission boundaries. Verified all 9 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
