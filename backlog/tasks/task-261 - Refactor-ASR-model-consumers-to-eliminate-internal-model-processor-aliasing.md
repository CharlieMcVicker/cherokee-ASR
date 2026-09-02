---
id: TASK-261
title: Refactor ASR model consumers to eliminate internal model/processor aliasing
status: Done
assignee:
  - '@agent-refactor'
created_date: '2026-09-02 15:01'
updated_date: '2026-09-02 15:13'
labels:
  - code-smell
  - refactor
  - models
dependencies: []
priority: medium
ordinal: 263000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code smell identified in PR #3 review: consumers instantiate CherokeeASRModel but unpack internal attributes (`model = asr_model.model`, `processor = asr_model.processor`, `device = asr_model.device`).

Target files:
1. `transcription/inference/batch.py`: Replace local aliasing of `model`, `processor`, `device` with direct calls to `asr_model.model`, `asr_model.processor`, `asr_model.device`, or `asr_model.to(device)`.
2. `transcription/utils/evaluation.py`: Replace unpacking of `asr_model.model` and `asr_model.processor` where `CherokeeASRModel` instance can be passed and used directly with `asr_model.to(device)`.
3. `transcription/inference/single.py`: Audit usage of `asr_model`.

Deliverables:
- Eliminate redundant local variable aliasing across model consumers.
- Use `asr_model.to(device)` instead of manual `asr_model.model.to(device)` and `asr_model.device = ...`.
- Ensure all 100 unit tests pass and pyright reports 0 errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audit all references to asr_model.model, asr_model.processor, and asr_model.device across transcription/
- [x] #2 Refactor evaluation and inference modules to rely on CherokeeASRModel abstractions or explicit parameter passing
- [x] #3 Ensure consistency across batch inference and evaluation pipelines
- [x] #4 Verify all tests pass and pyright typechecking succeeds
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refactor `transcription/inference/batch.py` to use `asr_model` directly without local aliases.
2. Refactor `transcription/utils/evaluation.py` evaluation loops to use `CherokeeASRModel.to()` and direct methods.
3. Verify backwards compatibility across test suites.
4. Run `pytest` and `pyright transcription`.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored ASR model consumers across inference and evaluation modules to eliminate internal model, processor, and device local variable aliasing. In batch.py and evaluation.py, references directly utilize CherokeeASRModel encapsulation (asr_model.processor, asr_model.device, asr_model.model, and asr_model.to('cpu') / asr_model.to(device)). Verified full unit test suite (106 tests passing) and pyright (0 errors).
<!-- SECTION:FINAL_SUMMARY:END -->
