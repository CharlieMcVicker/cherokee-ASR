---
id: TASK-259
title: 'Address code review feedback on PR #3'
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-02 15:01'
updated_date: '2026-09-02 15:03'
labels:
  - code-review
  - refactor
dependencies: []
priority: high
ordinal: 261000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address specific feedback from code review on PR #3: remove in-function imports in pipeline.py, remove model/processor/device aliasing in batch_inference_aligner.py, and remove syllabary field from language-agnostic ASRResult in asr_model.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Move in-function import in transcription/new_testament/pipeline.py to module top level
- [x] #2 Remove redundant model/processor/device aliasing in transcription/syllabary_enrichment/batch_inference_aligner.py
- [x] #3 Remove syllabary attribute from ASRResult and keep ASR models language-agnostic
- [x] #4 Verify all unit tests pass and pyright reports 0 errors
- [x] #5 Submit response to PR #3 review
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move in-function imports in transcription/new_testament/pipeline.py to top-level.
2. Remove model/processor/device aliasing in transcription/syllabary_enrichment/batch_inference_aligner.py and streamline batch inference.
3. Remove syllabary field from ASRResult in transcription/models/asr_model.py and update test_asr_model.py.
4. Run pytest and pyright to ensure zero regressions.
5. Post response comments on PR #3.
6. Commit changes and push to dev branch.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all code review feedback on PR #3:
1. Moved in-function imports in transcription/new_testament/pipeline.py to top-level module scope.
2. Removed local model/processor/device aliasing in transcription/syllabary_enrichment/batch_inference_aligner.py, added CherokeeASRModel.to(device) method, and cleaned up in-function cast/csv imports.
3. Removed syllabary field from ASRResult dataclass and decode() method in transcription/models/asr_model.py, keeping core ASR acoustic modeling language-agnostic.
4. Updated test_asr_model.py to test the revised ASRResult schema and CherokeeASRModel.to() method.
5. Verified 100/100 tests pass via pytest and pyright reports 0 errors.
6. Replied directly to all PR #3 review comment threads and posted a summary response on PR #3.
7. Scoped three new Backlog tasks (TASK-260, TASK-261, TASK-262) to address code smells and patterns across the broader codebase.
<!-- SECTION:FINAL_SUMMARY:END -->
