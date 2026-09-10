---
id: TASK-284
title: Realign Bible with Toneless Model and Calibrate Alignment Threshold
status: Done
assignee:
  - '@agent'
created_date: '2026-09-10 20:34'
updated_date: '2026-09-10 20:58'
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
- [x] #1 Run scripts/realign_bible.py across Mark and Matthew using CachedASREmissionsExtractor and updated confusion_cost_matrix_prebible.json
- [x] #2 Verify zero-latency emissions extraction cache hits across all 52 chapters
- [x] #3 Export updated split audio, per-book alignment records, combined bible_alignment_records.json, and training CSVs
- [x] #4 Execute scripts/find_alignment_threshold.py over the realigned dataset to calibrate optimal cost threshold T*
- [x] #5 Export updated runs/evaluation/alignment_threshold.json and evaluate yield/filter statistics
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Execute scripts/realign_bible.py across Mark and Matthew using CachedASREmissionsExtractor with toneless model charliemcvicker/length-only-20260704-155307-asr-cherokee-colon:76e62140955f4738abdab345ea34068b02d8d2a2 and runs/evaluation/confusion_cost_matrix_prebible.json.
2. Verify zero-latency emissions extraction cache hits across all chapters.
3. Export updated split audio wavs, mark_alignment_records.json, matthew_alignment_records.json, combined bible_alignment_records.json, and training CSVs.
4. Execute scripts/find_alignment_threshold.py over the realigned dataset to calibrate optimal cost threshold T*.
5. Export updated runs/evaluation/alignment_threshold.json and evaluate yield/filter statistics across Mark and Matthew.
6. Verify all test suites pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Realigned New Testament books Mark (16 chapters, 678 verses) and Matthew (28 chapters, 1,071 verses) using the toneless ASR model (charliemcvicker/length-only-20260704-155307-asr-cherokee-colon:76e62140955f4738abdab345ea34068b02d8d2a2) with pre-cached zero-latency emissions and the calibrated pre-Bible confusion cost matrix (runs/evaluation/confusion_cost_matrix_prebible.json). Generated updated 16kHz mono audio slices in cherokee_new_testament/split_audio/, per-book alignment manifests (mark_alignment_records.json, matthew_alignment_records.json), combined bible_alignment_records.json (1,749 verses), and training CSVs (mark.csv, matthew.csv). Calibrated alignment threshold T* = 0.073217 via interactive binary search and exported runs/evaluation/alignment_threshold.json. All 230 test cases pass.
<!-- SECTION:FINAL_SUMMARY:END -->
