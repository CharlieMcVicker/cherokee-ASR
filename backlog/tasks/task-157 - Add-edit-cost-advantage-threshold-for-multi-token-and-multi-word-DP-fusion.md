---
id: TASK-157
title: Add edit cost advantage threshold for multi-token and multi-word DP fusion
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 15:21'
updated_date: '2026-07-23 15:22'
labels: []
dependencies: []
ordinal: 153000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent unnecessary fusion when 1-to-1 alignments already match well. Only permit multi-token or multi-word fusion if concatenated edit cost is significantly lower than individual 1-to-1 match costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add fusion penalty/threshold to prevent merging well-matched 1-to-1 tokens
- [x] #2 Add unit test for false fusion prevention case
- [x] #3 All timestamping unit tests pass
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Calibrated DP fusion penalties in aligner.py so that multi-token/word fusion only occurs when concatenating yields a clear edit distance improvement over 1-to-1 matches. Added test_distinct_words_do_not_fuse test case.
<!-- SECTION:FINAL_SUMMARY:END -->
