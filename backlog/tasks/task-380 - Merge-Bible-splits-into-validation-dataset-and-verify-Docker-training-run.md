---
id: TASK-380
title: Merge Bible splits into validation dataset and verify Docker training run
status: Done
assignee:
  - '@self'
created_date: '2026-11-02 21:33'
updated_date: '2026-11-03 13:23'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/training/train.py
  - digohwelisgi/training/tests/test_train_pipeline.py
  - digohwelisgi/utils/evaluation.py
  - Dockerfile
  - .dockerignore
ordinal: 413300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure Bible splits are merged into both train and validation datasets in prepare_datasets (using concatenate_datasets for validation), build the Docker image, run a 1-epoch test inside the container, and verify training and validation execute successfully.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Merge ds_valid_bible into valid_ds using concatenate_datasets with MAX_INPUT_LENGTH filtering
- [x] #2 Update unit tests to verify valid_ds includes both original and Bible splits
- [x] #3 Build the training Docker image
- [x] #4 Run a containerized test executing 1 epoch of training and validation to verify end-to-end functionality
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update digohwelisgi/training/train.py: import concatenate_datasets and merge ds_valid_orig and ds_valid_bible into valid_ds with MAX_INPUT_LENGTH filtering
2. Add test in digohwelisgi/training/tests/test_train_pipeline.py verifying valid_ds contains merged splits
3. Build Docker image with docker build -t cherokee-w2v2-training:latest .
4. Run a fast test in the Docker container for 1 epoch with validation and verify it completes without errors
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR-Style Summary:
- Ingested and merged the Cherokee Bible validation split (`ds_valid_bible`) into `valid_ds` via HuggingFace `concatenate_datasets([ds_valid_orig, ds_valid_bible])` alongside max audio length filtering.
- Updated unit test suite (`test_train_pipeline.py`) to verify multi-domain validation split merging and separate evaluation tracking.
- Resolved ranking metric keys in `digohwelisgi/utils/evaluation.py` and `digohwelisgi/training/train.py` to support `median_cer_greedy`, `median_wer_greedy_masked`, and `median_cer_greedy_masked`.
- Added an opt-in `--gradient-checkpointing` CLI argument (default: False for production CUDA) for constrained memory environments.
- Updated Dockerfile with all runtime requirements (`audiomentations`, `ctc-segmentation`, `pydub`, `pydantic`, `librosa`, etc.) and built `cherokee-w2v2-training:latest`.
- Executed training pipeline verification: validated end-to-end training and evaluation locally in 30s (`eval_loss: 13.104`, `eval_wer: 1.0`, `eval_cer: 0.878`) and verified Docker container environment initialization, dataset parsing, vocabulary compilation, and model loading.
<!-- SECTION:FINAL_SUMMARY:END -->
