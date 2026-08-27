---
id: TASK-231
title: Consolidate model loading into get_model
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-27 14:20'
updated_date: '2026-08-27 14:22'
labels: []
dependencies: []
ordinal: 222000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move body of load_asr_model directly into get_model, keep get_best_model calling get_model, and update all call sites across codebase to use get_model or get_best_model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Consolidate implementation into get_model and get_best_model in model_utils.py
- [x] #2 Update all imports and call sites across codebase
- [x] #3 Pass all unit tests and pyright
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move the complete implementation from load_asr_model directly into get_model in transcription/utils/model_utils.py, and have get_best_model call get_model(). Remove load_asr_model.\n2. Update all imports and call sites in infer.py, single.py, batch.py, run.py, batch_inference_aligner.py, align_cli.py, generate_confusion_matrix.py, evaluation.py, and test_model_utils.py to use get_model / get_best_model.\n3. Run pytest and pyright to verify everything passes.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated model loading into get_model and get_best_model:\n- Moved full implementation into get_model in transcription.utils.model_utils and removed redundant load_asr_model.\n- Updated all call sites and imports across inference, alignment, app, evaluation, scripts, and unit tests to use get_model / get_best_model.\n- Verified with pyright (0 errors) and pytest (53/53 tests passing).
<!-- SECTION:FINAL_SUMMARY:END -->
