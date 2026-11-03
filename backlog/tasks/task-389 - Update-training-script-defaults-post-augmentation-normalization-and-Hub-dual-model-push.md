---
id: TASK-389
title: >-
  Update training script defaults, post-augmentation normalization, and Hub
  dual-model push
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 20:25'
updated_date: '2026-11-03 20:30'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/cherokee/training/train.py
  - digohwelisgi/cherokee/training/tests/test_train_pipeline.py
ordinal: 422300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare train.py for Vast.ai training runs by updating epoch/eval step defaults, re-normalizing audio post-augmentation in the CTC data collator, adding configurable hub_strategy, and pushing/tagging both the best validation model and promoted test_orig model along with evaluation logs to Hugging Face Hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CONFIG defaults updated (epochs: 30, eval_steps: 200, save_steps: 400, hub_strategy: all_checkpoints, hub_prefix: length-only) and CLI flags added
- [x] #2 DataCollatorCTCWithPadding re-normalizes waveform to zero mean and unit variance post-augmentation
- [x] #3 TrainingArguments configures hub_strategy from CONFIG
- [x] #4 train_model pushes combined validation winner to Hugging Face Hub and tags commit as best-combined-valid
- [x] #5 promote_best_checkpoint guards against SameFileError and main uploads promoted model and eval_logs to Hub
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update CONFIG and parse_args with new defaults and --hub-strategy, --hub-prefix flags.
2. Update DataCollatorCTCWithPadding with post-augmentation zero-mean unit-variance audio normalization.
3. Pass hub_strategy to TrainingArguments in initialize_model_and_trainer.
4. Update train_model to push best combined-validation model and tag best-combined-valid.
5. Guard promote_best_checkpoint against SameFileError and upload promoted model and eval_logs in main().
6. Run tests and typechecker to verify zero regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
### Summary of Changes
- Updated `CONFIG` defaults in `train.py`: `epochs=30`, `eval_steps=200`, `save_steps=400`, `hub_strategy="all_checkpoints"`, and `hub_prefix="length-only"`.
- Added `--hub-strategy` and `--hub-prefix` CLI flags to `parse_args()` with dynamic repo prefix generation.
- Added post-augmentation audio re-normalization in `DataCollatorCTCWithPadding` to ensure zero-mean and unit-variance before feeding the frozen feature encoder.
- Passed `hub_strategy` into `TrainingArguments` in `initialize_model_and_trainer`.
- Updated `train_model` to push the best combined-validation checkpoint and tag it as `best-combined-valid` on Hugging Face Hub.
- Guarded `promote_best_checkpoint` against `shutil.SameFileError` when the promoted checkpoint is "final".
- Added end-of-run Hub upload in `main()` for the promoted `test_orig` checkpoint and `eval_logs/` results summary.
- Added comprehensive unit tests in `test_train_pipeline.py` verifying all 4 updates.

### Verification
- `pytest digohwelisgi/cherokee/training/tests/test_train_pipeline.py` passed (10/10 tests).
- `pyright digohwelisgi` passed with 0 errors and 0 warnings.
<!-- SECTION:FINAL_SUMMARY:END -->
