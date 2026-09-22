---
id: TASK-360.6
title: >-
  Phase 1C: Implement ModelOutput, standalone inference procedures & cached
  ASRModel wrapper
status: Done
assignee:
  - '@phase-1c-implementor'
created_date: '2026-09-21 20:32'
updated_date: '2026-09-22 14:59'
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
- [x] #1 ModelOutput dataclass created in transcription.core.models.output wrapping lpz, vocab, decode_greedy(), decode_tokens(), and .save()/.load() .npz caching
- [x] #2 Standalone procedures infer_emissions() and infer_emissions_batch() implemented in transcription.core.models.inference consuming raw model/processor with optional cache_dir
- [x] #3 CherokeeASRModel / ASRModel exposes clean infer() and infer_batch() returning ModelOutput with zero pass-through forwarding wrappers
- [x] #4 Unit tests verify inference parity and caching behavior with pyright reporting 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement ModelOutput universal rich container in transcription.core.models.output with lpz, vocab, decode_greedy, decode_tokens, and save/load npz methods
2. Implement standalone inference procedures infer_emissions and infer_emissions_batch in transcription.core.models.inference with .npz caching
3. Implement generic ASRModel wrapper in transcription.core.models.model exposing clean infer() and infer_batch() returning ModelOutput
4. Update CherokeeASRModel to inherit from ASRModel while preserving existing procedural inference APIs
5. Add unit tests in transcription/core/models/tests/test_models.py and verify zero pyright errors and passing test suite
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ModelOutput universal data container in transcription.core.models.output with lpz emissions, vocab mapping, decode_greedy(), decode_tokens(), and compressed .npz save/load serialization. Implemented standalone inference procedures infer_emissions() and infer_emissions_batch() in transcription.core.models.inference supporting optional on-disk .npz caching and batch OOM handling. Created generic ASRModel wrapper in transcription.core.models.model exposing clean infer() and infer_batch() without pass-through delegation. Updated CherokeeASRModel to inherit from ASRModel while maintaining backwards compatibility with existing procedural layers. Added comprehensive unit tests in transcription/core/models/tests/test_models.py verifying inference parity, greedy decoding, and caching behavior. Verified with 27 passing model tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
