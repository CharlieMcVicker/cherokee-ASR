---
id: TASK-308
title: >-
  Add minimum character confidence thresholding to CTCSegmentationAligner
  anomaly detection
status: Done
assignee:
  - '@agent'
created_date: '2026-09-12 20:01'
updated_date: '2026-09-14 13:40'
labels: []
dependencies: []
ordinal: 324000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade CTCSegmentationAligner word anomaly detection to incorporate per-character minimum acoustic probability alongside geometric mean confidence, ensuring single-letter transcript typos like Mark 1:1 yihstv for ohstv are reliably flagged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add per-character minimum acoustic probability check or calibrate word confidence threshold in CTCSegmentationAligner
- [x] #2 Verify Mark 1:1 yihstv typo is marked flagged=True
- [x] #3 Verify false positive anomaly rate on 100-verse benchmark remains low
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added min_char_confidence to WordInterval domain model and CTCSegmentationAligner anomaly detection. Aligner now computes per-character minimum acoustic posterior alongside geometric mean word confidence. When min_char_confidence is below threshold (default 0.005), words like Mark 1:1 'yihstv' typo are reliably flagged with flagged=True. Verified with unit tests in test_ctc_aligner.py and full test suite passing.
<!-- SECTION:FINAL_SUMMARY:END -->
