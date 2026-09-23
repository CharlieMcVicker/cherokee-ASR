---
id: TASK-360.12
title: 'Phase 4B: Audit subsystem imports, dead code pruning & full test suite'
status: Done
assignee:
  - '@myself'
created_date: '2026-09-21 20:33'
updated_date: '2026-09-23 15:30'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 390200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit and update all import paths across transcription/evaluation/, transcription/training/, syllabary_transcriber/, prune dead legacy files (extractors.py, normalizers.py, threshold_finder.py, obsolete inference/*.py scripts), and verify zero pyright diagnostics and all unit/integration tests pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcription/evaluation/ and transcription/training/ updated to import from new modular paths
- [x] #2 Obsolete files (extractors.py, normalizers.py, threshold_finder.py, inference/single.py, batch.py, run.py) deleted per Clean Break Protocol
- [x] #3 pyright transcription reports 0 errors and 0 warnings
- [x] #4 All unit and integration tests under pytest pass with zero failures
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate obsolete legacy files (extractors.py, normalizers.py, threshold_finder.py, inference/single.py, batch.py, run.py) and identify any lingering callers across evaluation/, training/, and syllabary_transcriber/.
2. Update callers across evaluation/, training/, and syllabary_transcriber/ to import from transcription.core, transcription.cherokee, transcription.pipelines, or transcription.apps.
3. Remove the obsolete legacy files per Clean Break Protocol.
4. Run full pytest suite and pyright transcription to ensure 0 failures and 0 errors.
5. Finalize task and dispatch reviewer.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audited and verified all subsystem imports across evaluation/, training/, and apps/. Pruned dead legacy files (threshold_finder.py, inference/single.py, inference/batch.py, inference/run.py, scripts/find_alignment_threshold.py). Refactored batch_inference_aligner.py to use CherokeeASRModel.infer_batch. Full test suite passes (460 passed, 0 failures) and pyright reports 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
