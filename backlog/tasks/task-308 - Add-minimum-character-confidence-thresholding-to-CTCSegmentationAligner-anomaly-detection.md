---
id: TASK-308
title: >-
  Add minimum character confidence thresholding to CTCSegmentationAligner
  anomaly detection
status: To Do
assignee: []
created_date: '2026-09-12 20:01'
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
- [ ] #1 Add per-character minimum acoustic probability check or calibrate word confidence threshold in CTCSegmentationAligner
- [ ] #2 Verify Mark 1:1 yihstv typo is marked flagged=True
- [ ] #3 Verify false positive anomaly rate on 100-verse benchmark remains low
<!-- AC:END -->
