---
id: TASK-319
title: >-
  Calibrate per-token intrusion penalties and min-logprobs on Mark Chapter 1 and
  benchmark comparison
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 13:09'
updated_date: '2026-09-14 13:45'
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
- [x] #1 Run calibration grid search on Mark Chapter 1
- [x] #2 Evaluate diff categories with benchmark_ctc_segmentation_100_verses.py
- [x] #3 Verify recovery of authentic glottal stops and suppression of spurious breath noise
- [x] #4 Document optimal parameter set in docs/alignment.md
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create calibration script to grid search across /h/ penalties (3.0, 4.0, 4.5, 5.0), /'/ penalties (0.5, 0.8, 1.0, 1.2), and min-logprob settings on Mark Chapter 1.
2. Run benchmark_ctc_segmentation_100_verses.py with candidate configurations and evaluate diff categories (glottal_stop_added, aspiration_added, vowel_syncope_dropped).
3. Confirm optimal parameter configuration for authentic laryngeal recovery without spurious insertions.
4. Update docs/alignment.md with optimal parameter set and default configurations.
5. Verify tests and mark TASK-319 Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Executed calibration grid search across candidate penalties on Mark Chapter 1 (runs/evaluation/calibration_mark_01.json), ran 100-verse benchmark with calibrated parameters (runs/evaluation/ctc_segmentation_100_verses_comparison.json), and documented recommended defaults in docs/alignment.md.
<!-- SECTION:FINAL_SUMMARY:END -->
