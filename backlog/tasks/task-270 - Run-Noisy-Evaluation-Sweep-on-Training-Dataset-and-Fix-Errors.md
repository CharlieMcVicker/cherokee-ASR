---
id: TASK-270
title: Run Noisy Evaluation Sweep on Training Dataset and Fix Errors
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-02 17:11'
updated_date: '2026-09-02 17:37'
labels:
  - evaluation
  - asr
  - benchmark
dependencies: []
priority: high
type: task
ordinal: 282000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute scripts/run_noisy_eval.py on training_data/processed/cim-wav2vec2-train.csv across all noise and SNR tiers. Debug and fix any issues (path resolution, MPS/device compatibility, memory leaks, NaN values) until the entire run completes and produces all evaluation artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Execute run_noisy_eval.py on cim-wav2vec2-train.csv in cherokee-asr conda environment
- [x] #2 Fix any runtime errors, audio loading issues, or pipeline failures
- [x] #3 Verify all artifacts are successfully generated (eval_records.jsonl, confusion_matrix.csv, confusion_cost_matrix.json, visualization plots)
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Full training set run completes successfully without crash
- [x] #2 All 6 evaluation artifacts generated and verified
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check training_data/processed/cim-wav2vec2-train.csv and inspect audio path formats.
2. Inspect scripts/run_noisy_eval.py and evaluator.py for potential batch or memory issues on 3749 samples.
3. Launch run_noisy_eval.py and monitor execution.
4. Fix any encountered bugs or bottlenecks.
5. Verify generated artifacts.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully completed full noisy evaluation sweep on training_data/processed/cim-wav2vec2-train.csv (37,340 evaluations across 10 noise/SNR conditions). Pre-flight and runtime fixes resolved python environment auto-reexec, 1D audio tensor routing in get_probabilities, and audio preloading optimizations. All 6 artifacts generated and verified.
<!-- SECTION:FINAL_SUMMARY:END -->
