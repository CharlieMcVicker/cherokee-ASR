---
id: TASK-361
title: Prune ASREmissionsExtractor hierarchy and extractors.py
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-23 15:55'
updated_date: '2026-09-23 16:04'
labels: []
dependencies: []
modified_files:
  - transcription/alignment/extractors.py
  - transcription/alignment/tests/test_extractors.py
  - transcription/apps/cli.py
  - transcription/pipelines/scripture/pipeline.py
  - transcription/pipelines/dialogue/pipeline.py
  - scripts/batch_cache_emissions.py
  - transcription/alignment/__init__.py
  - transcription/core/models/inference.py
  - transcription/core/models/model.py
  - transcription/alignment/tests/test_cli.py
  - transcription/alignment/tests/test_syllabary_runners.py
  - transcription/alignment/tests/test_interview_realignment.py
  - transcription/pipelines/dialogue/tests/test_dialogue_pipeline.py
  - transcription/new_testament/tests/test_pipeline.py
ordinal: 391300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Legacy ASREmissionsExtractor hierarchy, CherokeeASRExtractor, CachedASREmissionsExtractor, and extractors.py remain in transcription/alignment/ as backward-compatibility shims despite ModelOutput and on-model .npz caching being established in Tier 1. Remove extractors.py, its unit tests, and update callers (scripture pipeline, apps/cli.py, batch_cache_emissions.py) to use ModelOutput directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 extractors.py and test_extractors.py deleted
- [x] #2 transcription.apps.cli and transcription.pipelines.scripture updated to use ModelOutput
- [x] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect all files referencing extractors.py and ASREmissionsExtractor hierarchy.\n2. Update callers (transcription/apps/cli.py, transcription/pipelines/scripture/pipeline.py, transcription/pipelines/dialogue/pipeline.py, scripts/batch_cache_emissions.py, transcription/new_testament/tests/test_pipeline.py, and tests with patch('transcription.alignment.extractors.segment_long_audio')) to use ModelOutput, infer_emissions, and transcription.core.audio directly.\n3. Delete transcription/alignment/extractors.py and transcription/alignment/tests/test_extractors.py.\n4. Clean up imports in transcription/alignment/__init__.py and docs/alignment.md if needed.\n5. Run pytest and pyright to ensure full test suite passes with zero errors.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned legacy ASREmissionsExtractor hierarchy (ASREmissionsExtractor, CherokeeASRExtractor, CachedASREmissionsExtractor, CallbackEmissionsExtractor, PrecomputedEmissionsExtractor) and deleted extractors.py along with test_extractors.py. Refactored all callers across transcription/apps/cli.py, transcription/pipelines/scripture/pipeline.py, transcription/pipelines/dialogue/pipeline.py, and scripts/batch_cache_emissions.py to use ModelOutput universal currency, infer_emissions, infer_emissions_batch, and CherokeeASRModel directly. Verified 447 tests passing in pytest and 0 errors in pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
