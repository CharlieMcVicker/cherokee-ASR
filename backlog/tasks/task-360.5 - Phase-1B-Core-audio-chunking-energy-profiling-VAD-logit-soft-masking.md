---
id: TASK-360.5
title: 'Phase 1B: Core audio chunking, energy profiling & VAD logit soft-masking'
status: Done
assignee:
  - '@phase-1b-implementor'
created_date: '2026-09-21 20:32'
updated_date: '2026-09-22 14:51'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 387200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract language-agnostic audio segmentation, energy profiling, and Silero VAD logit soft-masking into transcription.core.audio, eliminating hardcoded Cherokee paths or settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 segment_long_audio, get_energy_profile, and AudioChunk reside in transcription.core.audio.segment
- [x] #2 Silero VAD logit masking functions (apply_vad_soft_masking, mask_non_speech_logits, extract_vad_intervals) reside in transcription.core.audio.masking
- [x] #3 Audio unit tests (test_audio_segmenter.py, test_non_speech_masking.py) pass under new imports and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/core/__init__.py and transcription/core/audio/__init__.py
2. Implement transcription.core.audio.segment with AudioChunk, get_energy_profile, segment_audio_from_profile, compute_metrics, get_best_parameters, split_long_segments_smart, and segment_long_audio
3. Implement transcription.core.audio.masking with SileroVADDetector, _load_audio_as_16k_tensor, mask_non_speech_logits, apply_vad_soft_masking (alias/entrypoint for mask_non_speech_logits), and extract_vad_intervals
4. Re-export segment and masking in transcription/core/audio/__init__.py
5. Update or re-export in transcription/audio/ for seamless compatibility and update callers across the codebase to direct transcription.core.audio imports
6. Update audio unit tests (test_audio_segmenter.py and test_non_speech_masking.py) to test transcription.core.audio
7. Verify with pytest tests and pyright transcription
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented transcription.core.audio package extracting language-agnostic audio segmentation, energy profiling, and Silero VAD logit soft-masking:
- transcription.core.audio.segment provides AudioChunk, get_energy_profile, segment_audio_from_profile, compute_metrics, get_best_parameters, split_long_segments_smart, and segment_long_audio.
- transcription.core.audio.masking provides SileroVADDetector, mask_non_speech_logits, apply_vad_soft_masking, and extract_vad_intervals.
- transcription.audio re-exports core functionality for backward-compatibility.
- Verified test_audio_segmenter.py and test_non_speech_masking.py passing (7 passed), alignment test suite passing (40 passed), and pyright transcription reporting 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
