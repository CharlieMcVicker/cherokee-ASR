---
id: TASK-229
title: >-
  Create get_model and get_best_model functions and normalize model
  instantiation
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-27 14:09'
updated_date: '2026-08-27 14:16'
labels: []
dependencies: []
ordinal: 220000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement unified get_model(path_or_repo, revision=...) and get_best_model() helper functions in transcription.utils.model_utils and normalize Wav2Vec2 model/processor instantiation across the codebase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement get_model and get_best_model in transcription.utils.model_utils
- [x] #2 Normalize model and processor loading across inference, alignment, and evaluation scripts
- [x] #3 Ensure unit tests and pyright pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement get_model, get_best_model, and load_asr_model in transcription/utils/model_utils.py with token, device, caching, and custom processor support.\n2. Re-export get_model, get_best_model, and load_asr_model from transcription/inference/infer.py.\n3. Normalize all non-training model instantiations across inference (single.py, batch.py, run.py), alignment (align_cli.py, batch_inference_aligner.py), scripts (generate_confusion_matrix.py), and evaluation (transcription/utils/evaluation.py).\n4. Run pyright and pytest to verify all tests and type checks pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized Wav2Vec2 model and processor instantiation across the codebase:\n- Implemented canonical load_asr_model, get_model, and get_best_model in transcription.utils.model_utils with token auto-resolution, device detection, and cache control.\n- Re-exported functions from transcription.inference.infer.\n- Updated single.py, batch.py, run.py, batch_inference_aligner.py, align_cli.py, generate_confusion_matrix.py, and evaluation.py to use the unified loader.\n- Added comprehensive unit tests in transcription/utils/test_model_utils.py.\n- Verified all 53 unit tests and pyright type checks pass cleanly with 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
