---
id: TASK-403
title: 'Phase 5: Multi-Task Training Pipeline, Evaluation Harness & Local Smoke Gate'
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:25'
updated_date: '2026-11-08 04:04'
labels: []
dependencies:
  - TASK-402
modified_files:
  - digohwelisgi/cherokee/training/train.py
ordinal: 436300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire Wav2Vec2ForDualCTC, 50/50 interleaved sampling, and multi-task evaluation metrics (CV WER/CER, Tone TER, Reconciled CER) into train.py and evaluation scripts, and pass the local MPS smoke test gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 train.py supports dual-head training with configurable --tone-loss-weight, --num-branch-layers, and --vocab-tone-path
- [x] #2 Multi-task evaluation computes CV WER/CER, Tone TER on tone-annotated rows, and Reconciled CER, selecting best model by tone_cer
- [x] #3 5-step local smoke test on Mac MPS/CPU completes forward/backward passes, optimizer updates, evaluation, and dual-tokenizer checkpoint saving with 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update digohwelisgi/cherokee/training/train.py for dual-head multi-task training, CLI args, dual vocabularies, dual dataset encoding, and compute_dual_metrics.
2. Update digohwelisgi/utils/evaluation.py and digohwelisgi/cherokee/training/evaluate_local_checkpoints.py for dual-head checkpoint evaluation.
3. Add digohwelisgi/cherokee/training/tests/test_metrics_and_eval.py and run local 5-step smoke test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added --save-total-limit (default 3) and preserved exact --hub-model-id on resume.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated train.py, evaluation.py, and evaluate_local_checkpoints.py for dual-head multi-task training and evaluation (wer, cer, tone_cer, reconciled_cer) and passed the 5-step local MPS smoke test gate.
<!-- SECTION:FINAL_SUMMARY:END -->
