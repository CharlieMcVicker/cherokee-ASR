---
id: TASK-265
title: Implement unified model resolution and fallback factory on CherokeeASRModel
status: Done
assignee:
  - '@agent-model-factory'
created_date: '2026-09-02 15:33'
updated_date: '2026-09-02 15:35'
labels:
  - refactor
  - models
  - enhancement
dependencies: []
priority: medium
ordinal: 267000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement CherokeeASRModel.from_pretrained_or_best(path_or_repo: Optional[str] = None, revision: Optional[str] = None, token: Optional[str] = None, fallback_repo: str = 'facebook/wav2vec2-base-960h', **kwargs) in transcription/models/asr_model.py.

This factory method handles nullable paths and automatic fallbacks to best_model.json or public base models cleanly without boilerplate try/except in caller scripts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add from_pretrained_or_best classmethod to CherokeeASRModel
- [x] #2 Refactor CLI and pipeline callers to use from_pretrained_or_best
- [x] #3 Add unit tests in test_asr_model.py covering fallback and config resolution
- [x] #4 Verify all unit tests pass and pyright reports 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented CherokeeASRModel.from_pretrained_or_best factory and unit tests
<!-- SECTION:FINAL_SUMMARY:END -->
