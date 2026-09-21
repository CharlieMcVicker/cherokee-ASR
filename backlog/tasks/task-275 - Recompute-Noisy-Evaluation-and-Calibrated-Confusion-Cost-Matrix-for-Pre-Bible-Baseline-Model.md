---
id: TASK-275
title: >-
  Recompute Noisy Evaluation and Calibrated Confusion Cost Matrix for Pre-Bible
  Baseline Model
status: Done
assignee:
  - '@subagent-275'
created_date: '2026-09-10 17:50'
updated_date: '2026-09-10 17:58'
labels:
  - evaluation
  - alignment
  - distance-metrics
dependencies: []
ordinal: 287000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run noisy ASR evaluation on the pre-Bible baseline model charliemcvicker/asr-cherokee:5464d15 across validation audio to produce eval records. Generate calibrated operational confusion matrix CSV and substitution cost JSON artifact (confusion_cost_matrix_prebible.json) using ConfusionCostEngine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run noisy evaluation on charliemcvicker/asr-cherokee:5464d15 and generate runs/evaluation/eval_records_prebible.jsonl
- [x] #2 Accumulate unigram confusion matrix and Dirichlet-smoothed probabilities for pre-Bible model
- [x] #3 Export runs/evaluation/confusion_matrix_prebible.csv and runs/evaluation/confusion_cost_matrix_prebible.json
- [x] #4 Validate ConfusionMatrixCostMetric.from_json loads the new pre-Bible artifact cleanly
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All unit tests in transcription/alignment and transcription/evaluation pass
- [x] #2 Pre-Bible evaluation records, CSV matrix, and JSON cost artifact generated and verified
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recomputed noisy acoustic evaluation and empirical substitution confusion cost matrix for pre-Bible baseline model charliemcvicker/asr-cherokee@5464d15. Updated scripts/run_noisy_eval.py to support --revision and customizable output filenames/suffixes. Generated runs/evaluation/eval_records_prebible.jsonl (1,860 records across 10 SNR/noise conditions), runs/evaluation/confusion_matrix_prebible.csv, runs/evaluation/confusion_cost_matrix_prebible.json, and manifold plots. Verified ConfusionMatrixCostMetric cleanly loads the pre-Bible cost artifact and computes Cherokee phonetic distances. All unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
