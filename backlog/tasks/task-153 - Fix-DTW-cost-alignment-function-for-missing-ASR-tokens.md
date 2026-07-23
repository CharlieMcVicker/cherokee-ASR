---
id: TASK-153
title: Fix DTW cost alignment function for missing ASR tokens
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:42'
updated_date: '2026-07-23 14:42'
labels: []
dependencies: []
ordinal: 149000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix sliding window / DTW alignment in aligner.py to handle missing or dropped ASR tokens (e.g. dropped 'hi-a') and prevent greedy 1-off token shift misalignments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Needleman-Wunsch or token-level DTW edit alignment implemented in aligner.py
- [x] #2 Dropped ASR tokens leave empty/interpolated GT intervals rather than shifting subsequent words
- [x] #3 Tests pass for token edit alignment
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced proportional character range slicing with Needleman-Wunsch dynamic programming edit-distance alignment in _align_words_char_range.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Needleman-Wunsch dynamic programming alignment in transcription.timestamping.aligner. Replaced proportional string character distribution with DP matrix backtracking to align GT words to ASR emission tokens. When the model drops a word (such as 'hi-a'), DP cleanly marks the dropped GT word with zero-duration interpolation rather than shifting all subsequent words by 1. Verified all 9 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
