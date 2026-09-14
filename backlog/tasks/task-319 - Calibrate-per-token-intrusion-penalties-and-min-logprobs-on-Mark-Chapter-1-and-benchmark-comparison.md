---
id: TASK-319
title: >-
  Calibrate per-token intrusion penalties and min-logprobs on Mark Chapter 1 and
  benchmark comparison
status: To Do
assignee: []
created_date: '2026-09-14 13:09'
updated_date: '2026-09-14 13:29'
labels: []
dependencies:
  - TASK-318
  - TASK-308
ordinal: 335000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform calibration grid search across /h/ penalties (3.0 - 5.0), /'/ penalties (0.5 - 1.2), and min-logprobs (e.g. log(0.35) for h, log(0.20) for /'/) on Mark Chapter 1. Evaluate diffs with benchmark_ctc_segmentation_100_verses.py to verify genuine glottal stops and /h/ are recovered without spurious noise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Run calibration grid search on Mark Chapter 1
- [ ] #2 Evaluate diff categories with benchmark_ctc_segmentation_100_verses.py
- [ ] #3 Verify recovery of authentic glottal stops and suppression of spurious breath noise
- [ ] #4 Document optimal parameter set in docs/alignment.md
<!-- AC:END -->
