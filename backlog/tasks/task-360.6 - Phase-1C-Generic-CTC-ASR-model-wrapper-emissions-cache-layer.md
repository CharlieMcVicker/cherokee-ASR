---
id: TASK-360.6
title: >-
  Phase 1C: Implement ModelOutput, standalone inference procedures & cached
  ASRModel wrapper
status: To Do
assignee: []
created_date: '2026-09-21 20:32'
updated_date: '2026-09-21 20:49'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 387300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement ModelOutput dataclass and standalone inference procedures (infer_emissions, infer_emissions_batch) consuming raw Hugging Face models/processors with built-in optional .npz caching. CherokeeASRModel / ASRModel becomes a thin wrapper exposing only infer() and infer_batch() returning ModelOutput, eliminating all pass-through forwarding methods.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ModelOutput dataclass created in transcription.core.models.output wrapping lpz, vocab, decode_greedy(), decode_tokens(), and .save()/.load() .npz caching
- [ ] #2 Standalone procedures infer_emissions() and infer_emissions_batch() implemented in transcription.core.models.inference consuming raw model/processor with optional cache_dir
- [ ] #3 CherokeeASRModel / ASRModel exposes clean infer() and infer_batch() returning ModelOutput with zero pass-through forwarding wrappers
- [ ] #4 Unit tests verify inference parity and caching behavior with pyright reporting 0 errors
<!-- AC:END -->
