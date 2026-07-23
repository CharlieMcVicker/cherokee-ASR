---
id: TASK-154
title: Support multi-to-multi GT word fusion in Needleman-Wunsch DP alignment
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:45'
updated_date: '2026-07-23 14:45'
labels: []
dependencies: []
ordinal: 150000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor Needleman-Wunsch DP alignment in aligner.py to allow single ASR emission tokens (e.g., 'nvhskayohi\'a') to match and fuse multiple ground truth words ('nasgiya' + 'hi\'a').
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Merged ASR emission tokens correctly map to fused GT word sequences
- [x] #2 Tests pass for multi-word GT fusion in Needleman-Wunsch alignment
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refactored _align_words_char_range to compute DP transitions for fusing K consecutive GT words (1..4) onto single ASR emission tokens.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Supported multi-GT word fusion in Needleman-Wunsch DP alignment in transcription.timestamping.aligner. The DP state transitions now evaluate multi-word GT spans ('nasgiya' + 'hi\'a') against single ASR emission tokens ('nvhskayohi\'a'). When an ASR token merges multiple GT words, DP computes the edit cost against the concatenated GT text and maps the fused GT label onto the exact ASR token interval. Verified all 9 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
