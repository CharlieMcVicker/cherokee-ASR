---
id: TASK-261
title: Refactor ASR model consumers to eliminate internal model/processor aliasing
status: To Do
assignee: []
created_date: '2026-09-02 15:01'
labels:
  - code-smell
  - refactor
  - models
dependencies: []
ordinal: 263000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code smell identified in PR #3 review: consumers instantiate CherokeeASRModel but unpack internal attributes (model = asr_model.model, processor = asr_model.processor, device = asr_model.device). Audit all model consumers (batch inference, evaluation utils, etc.) to use CherokeeASRModel public methods and properties cleanly without local variable aliasing or internal leakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Audit all references to asr_model.model, asr_model.processor, and asr_model.device across transcription/
- [ ] #2 Refactor evaluation and inference modules to rely on CherokeeASRModel abstractions or explicit parameter passing
- [ ] #3 Ensure consistency across batch inference and evaluation pipelines
- [ ] #4 Verify all tests pass and pyright typechecking succeeds
<!-- AC:END -->
