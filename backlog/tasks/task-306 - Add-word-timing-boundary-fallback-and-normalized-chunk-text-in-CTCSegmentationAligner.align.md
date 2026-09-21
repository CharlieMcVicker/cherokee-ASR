---
id: TASK-306
title: >-
  Add word-timing boundary fallback and normalized chunk text in
  CTCSegmentationAligner.align
status: Done
assignee:
  - '@agent'
created_date: '2026-09-12 19:53'
updated_date: '2026-09-12 19:54'
labels: []
dependencies: []
ordinal: 322000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pass normalized chunk text to determine_utterance_segments and fallback to word intervals boundary when determine_utterance_segments produces empty or collapsed segments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pass normalized words string to determine_utterance_segments
- [x] #2 Fallback c_start and c_end to harvested word intervals when segment is collapsed
- [x] #3 Verify Mark chapter 1 realignment without dropped verses
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Passed normalized words string to determine_utterance_segments and added fallback to harvested word intervals when segment is collapsed, successfully realigning Mark Chapter 1 across all 45 verses.
<!-- SECTION:FINAL_SUMMARY:END -->
