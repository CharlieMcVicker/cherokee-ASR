---
id: TASK-283
title: >-
  Recompute Pre-Bible Noisy Evaluation and Calibrated Confusion Cost Matrix with
  Toneless Model
status: Done
assignee:
  - '@agent'
created_date: '2026-09-10 20:23'
updated_date: '2026-09-10 20:31'
labels:
  - evaluation
  - alignment
  - distance-metrics
dependencies: []
ordinal: 295000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run multi-SNR acoustic evaluation sweep on the toneless Cherokee ASR model checkpoint (charliemcvicker/length-only-20260704-155307-asr-cherokee-colon:76e62140955f4738abdab345ea34068b02d8d2a2) using scripts/run_noisy_eval.py to update the pre-Bible confusion and cost matrix artifacts (confusion_cost_matrix_prebible.json).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Execute noisy evaluation sweep on length-only-20260704-155307-asr-cherokee-colon producing eval_records_prebible.jsonl
- [x] #2 Accumulate character confusions and export confusion_matrix_prebible.csv
- [x] #3 Compute logarithmic substitution cost matrix and export confusion_cost_matrix_prebible.json
- [x] #4 Verify ConfusionMatrixCostMetric and PhonologicalConfusionCostMetric cleanly load updated pre-Bible artifact
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recomputed multi-SNR acoustic evaluation and calibrated empirical confusion cost matrix for toneless model charliemcvicker/length-only-20260704-155307-asr-cherokee-colon:76e62140955f4738abdab345ea34068b02d8d2a2. Generated runs/evaluation/eval_records_prebible.jsonl (1,850 records across 10 SNR/noise conditions), updated runs/evaluation/confusion_matrix_prebible.csv, and exported calibrated runs/evaluation/confusion_cost_matrix_prebible.json (17 unigram tokens without tone digits). Verified ConfusionMatrixCostMetric and PhonologicalConfusionCostMetric cleanly load the updated pre-Bible artifacts and compute distances. All unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
