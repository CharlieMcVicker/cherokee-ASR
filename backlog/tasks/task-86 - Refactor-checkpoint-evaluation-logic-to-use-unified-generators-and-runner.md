---
id: TASK-86
title: Refactor checkpoint evaluation logic to use unified generators and runner
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-02 21:09'
updated_date: '2026-07-02 21:10'
labels: []
dependencies: []
ordinal: 82000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate duplicated checkpoint evaluation code from train.py, evaluate_checkpoint.py, evaluate_local_checkpoints.py, and evaluate_revisions.py into a single evaluation module in transcription/utils/evaluation.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement model generators and run_evaluation in transcription/utils/evaluation.py
- [x] #2 Refactor train.py to use the new evaluation module
- [x] #3 Refactor evaluate_local_checkpoints.py, evaluate_revisions.py, and evaluate_checkpoint.py to use the new module
- [x] #4 Verify all refactored scripts run correctly without regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/utils/evaluation.py with generators and run_evaluation.\n2. Update transcription/training/train.py to import and use the new helper.\n3. Update evaluate_checkpoint.py, evaluate_local_checkpoints.py, and evaluate_revisions.py.\n4. Run a quick check/dry run of the scripts to verify syntax/imports.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored checkpoint evaluation logic by creating a unified module transcription/utils/evaluation.py. Implemented generator functions (local, hf, single) and a unified run_evaluation utility. Refactored train.py, evaluate_checkpoint.py, evaluate_local_checkpoints.py, and evaluate_revisions.py to consume the shared evaluation code, eliminating duplicate code and ensuring consistent metric computation across local and revision-based evaluation scripts.
<!-- SECTION:FINAL_SUMMARY:END -->
