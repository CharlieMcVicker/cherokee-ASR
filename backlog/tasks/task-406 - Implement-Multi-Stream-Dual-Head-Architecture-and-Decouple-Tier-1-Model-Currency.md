---
id: TASK-406
title: >-
  Implement Multi-Stream Dual-Head Architecture and Decouple Tier 1 Model
  Currency
status: Done
assignee:
  - '@self'
created_date: '2026-11-08 15:19'
updated_date: '2026-11-08 15:30'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/core/models/output.py
  - digohwelisgi/core/models/model.py
  - digohwelisgi/core/models/inference.py
  - digohwelisgi/core/models/__init__.py
  - digohwelisgi/core/__init__.py
  - digohwelisgi/cherokee/models/dual_head.py
  - digohwelisgi/cherokee/models/loader.py
  - digohwelisgi/cherokee/enrichment/syllable_alignment.py
  - digohwelisgi/cherokee/enrichment/__init__.py
  - digohwelisgi/core/models/tests/test_models.py
  - digohwelisgi/cherokee/tests/test_dual_inference_e2e.py
  - scripts/run_noisy_eval.py
ordinal: 439300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the multi-stream dual-head architecture plan from dual_head_architecture_plan.md. Refactor ModelOutput to support generic auxiliary streams, add VirtualHeadModel to ASRModel, isolate Cherokee tone reconciliation to Tier 2, and verify complete language-agnostic plug-and-play for single-head models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refactor ModelOutput to use auxiliary_streams: Dict[str, ModelOutput] with .tone and .get_stream() while preserving .npz serialization
- [x] #2 Remove Tier 2 Cherokee reconciliation imports from Tier 1 output.py and provide reconcile_model_output in digohwelisgi.cherokee
- [x] #3 Implement VirtualHeadModel and .as_head() in ASRModel, and add .segmental and .tone properties on CherokeeASRModel
- [x] #4 Update infer_emissions and Wav2Vec2ForDualCTC for multi-stream output and compute_tone bypass
- [x] #5 Verify all unit tests pass and Pyright reports 0 type errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refactor ModelOutput in digohwelisgi/core/models/output.py to support auxiliary_streams: Dict[str, ModelOutput], with backwards-compatible .tone, .lpz_tone, .vocab_tone properties, and remove Cherokee imports.
2. Add reconcile_model_output to digohwelisgi/cherokee/enrichment/ and CherokeeASRModel.decode_reconciled in digohwelisgi/cherokee/models/loader.py.
3. Implement VirtualHeadModel in digohwelisgi/core/models/model.py and add .as_head() to ASRModel, plus .segmental and .tone properties to CherokeeASRModel.
4. Update infer_emissions to populate auxiliary_streams and support compute_tone option.
5. Update unit tests in digohwelisgi/core/models/tests/, digohwelisgi/cherokee/tests/, and verify with pytest and pyright.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented multi-stream dual-head architecture: refactored ModelOutput to support generic auxiliary_streams (exposing .tone and .get_stream while preserving .npz serialization and backwards-compatible .lpz_tone/.vocab_tone properties), added VirtualHeadModel and .as_head() to ASRModel, exposed .segmental and .tone views on CherokeeASRModel, added compute_tone bypass to Wav2Vec2ForDualCTC, isolated Cherokee reconciliation into reconcile_model_output in digohwelisgi.cherokee.enrichment, fixed cv_text reference column in scripts/run_noisy_eval.py, and verified 409/409 pytest suite pass and 0 Pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
