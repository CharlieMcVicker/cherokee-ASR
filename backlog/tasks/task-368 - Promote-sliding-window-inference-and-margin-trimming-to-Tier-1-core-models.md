---
id: TASK-368
title: Promote sliding-window inference and margin trimming to Tier 1 core models
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-24 14:39'
updated_date: '2026-09-24 14:42'
labels: []
dependencies: []
modified_files:
  - transcription/core/models/inference.py
  - transcription/core/models/__init__.py
  - transcription/core/models/model.py
  - transcription/cherokee/models/loader.py
  - transcription/core/models/tests/test_models.py
ordinal: 398300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Overlapping sliding-window inference with margin trimming in CherokeeASRModel is purely language-agnostic acoustic temporal processing. Moving it to transcription.core.models enables all CTC models to perform long-form audio inference returning ModelOutput universal currency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement standalone infer_emissions_sliding_window procedure in transcription.core.models.inference returning ModelOutput
- [x] #2 Add infer_sliding_window method on ASRModel in transcription.core.models.model
- [x] #3 Update CherokeeASRModel and any callers/aligners to use the Tier 1 sliding-window engine
- [x] #4 Add unit tests for sliding-window inference, margin trimming, and edge cases in transcription.core.models.tests
- [x] #5 Verify pytest and pyright transcription pass with 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement infer_emissions_sliding_window in transcription/core/models/inference.py with audio slicing, margin trimming, logit concatenation, vocab extraction, caching, and chunk_seconds > 2 * margin_seconds validation.\n2. Add infer_sliding_window method to ASRModel in transcription/core/models/model.py returning ModelOutput.\n3. Update CherokeeASRModel.get_logits_sliding_window in transcription/cherokee/models/loader.py to delegate to self.infer_sliding_window(...).lpz converted to torch.Tensor.\n4. Add comprehensive unit tests in transcription/core/models/tests/test_models.py covering single chunk, multi-chunk with trimming, invalid chunk/margin validation, and caching.\n5. Run pytest and pyright to verify zero regressions and type safety.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Promoted sliding-window inference and margin trimming to Tier 1 core models:
1. Implemented standalone procedural infer_emissions_sliding_window in transcription/core/models/inference.py returning ModelOutput, with margin trimming, temporal stitching, parameter validation, and on-disk .npz caching.
2. Added infer_sliding_window method on ASRModel in transcription/core/models/model.py.
3. Updated CherokeeASRModel.get_logits_sliding_window to cleanly delegate to self.infer_sliding_window and convert output lpz to torch.Tensor.
4. Added comprehensive unit tests in transcription/core/models/tests/test_models.py verifying short audio single-pass, multi-chunk margin trimming, invalid chunk/margin validation, and caching.
5. Verified 100% test pass (399 tests) and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
