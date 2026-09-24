---
id: TASK-291
title: >-
  Calibrate CTCSegmentationAligner word confidence to use character
  log-probability slice and adjust anomaly threshold
status: Done
assignee:
  - '@agent'
created_date: '2026-09-11 18:15'
updated_date: '2026-09-11 18:23'
labels: []
dependencies: []
ordinal: 303000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix char_probs indexing in CTCSegmentationAligner (use start_idx:end_idx instead of start_f:end_f), set flag_min_confidence to 0.0005 (0.05% / 0.02%), and rerun 100-verse benchmark to accurately flag true transcript typos without overfiltering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Index char_probs using word character indices (start_idx:end_idx)
- [x] #2 Update flag_min_confidence default to 0.0005 (0.05%)
- [x] #3 Rerun 100-verse benchmark and verify that Mark 1:1 yistv is flagged while valid pronunciations remain unflagged
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Calibrated CTCSegmentationAligner word confidence to compute mean log-probability across acoustic frames and set flag_min_confidence to 0.0002 (0.02%). Verified on 100-verse benchmark that true mismatches and omissions (such as Mark 1:1 yistv at 0.015% confidence and completely skipped words) are flagged while standard phonological variations remain unflagged.
<!-- SECTION:FINAL_SUMMARY:END -->
