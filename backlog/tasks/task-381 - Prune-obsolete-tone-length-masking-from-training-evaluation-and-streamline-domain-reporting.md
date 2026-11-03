---
id: TASK-381
title: >-
  Prune obsolete tone/length masking from training evaluation and streamline
  domain reporting
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 14:00'
updated_date: '2026-11-03 14:12'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/training/evaluate_checkpoint.py
  - digohwelisgi/training/evaluate_local_checkpoints.py
  - digohwelisgi/training/evaluate_revisions.py
  - digohwelisgi/training/train.py
  - digohwelisgi/utils/evaluation.py
  - digohwelisgi/utils/test_evaluation.py
  - digohwelisgi/training/tests/test_train_pipeline.py
  - digohwelisgi/utils/test_model_utils.py
ordinal: 414300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Legacy evaluation code tracked vowel length and tone masked metrics (stripping colons and tones). Since current Cherokee ASR training targets are canonical toneless and lengthless TTH phonetics, these transformations are no-ops and create confusing dual unmasked/masked reporting. Prune legacy masking transforms from evaluation and train reporting, selecting best checkpoints based directly on aggregate WER and CER across Original and Bible test domain splits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prune legacy length/tone masked metrics and transforms from train.py and evaluation.py
- [x] #2 Streamline checkpoint ranking and promotion to standard WER and CER across Original and Bible domains
- [x] #3 Update unit tests to verify streamlined ranking without masked keys
- [x] #4 Verify all unit tests and pyright pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect evaluation.py and train.py for all references to strip_length, strip_tones, strip_both, and masked metrics
2. Refactor run_evaluation in evaluation.py to compute standard WER and CER directly on hypotheses without redundant transform variants
3. Refactor evaluate_checkpoints_dual and save_results_summary in train.py to rank and summarize by standard aggregate WER and CER
4. Update unit tests in test_train_pipeline.py and test_evaluation (if present)
5. Run pyright and pytest, commit changes, update PR #10, and mark TASK-381 Done
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR-Style Summary:
- Pruned obsolete vowel length and tone masking transforms (`strip_length`, `strip_tones`, `strip_both`) from `digohwelisgi/utils/evaluation.py` and evaluation driver scripts.
- Streamlined `run_evaluation` to produce clean, direct corpus WER and CER alongside sentence medians without redundant transform variants.
- Streamlined `evaluate_checkpoints_dual` and `save_results_summary` in `train.py` to rank and summarize directly by standard WER and CER across Original and Bible test domain splits.
- Updated all test suites and resolved runtime environment variable patch resolution in `test_model_utils.py`.
- Verified all 369 tests pass with zero regressions and pyright reports 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
