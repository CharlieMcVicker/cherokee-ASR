---
id: TASK-284
title: Realign Bible with Toneless Model and Calibrate Alignment Threshold
status: To Do
assignee: []
created_date: '2026-09-10 20:34'
labels:
  - alignment
  - new-testament
  - evaluation
dependencies: []
ordinal: 296000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute full audio-transcript realignment across New Testament books (Mark and Matthew) using the pre-cached toneless ASR emissions and updated confusion cost metric, then run scripts/find_alignment_threshold.py to determine the optimal alignment cost threshold T* and evaluate dataset yield.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Run scripts/realign_bible.py across Mark and Matthew using CachedASREmissionsExtractor and updated confusion_cost_matrix_prebible.json
- [ ] #2 Verify zero-latency emissions extraction cache hits across all 52 chapters
- [ ] #3 Export updated split audio, per-book alignment records, combined bible_alignment_records.json, and training CSVs
- [ ] #4 Execute scripts/find_alignment_threshold.py over the realigned dataset to calibrate optimal cost threshold T*
- [ ] #5 Export updated runs/evaluation/alignment_threshold.json and evaluate yield/filter statistics
<!-- AC:END -->
